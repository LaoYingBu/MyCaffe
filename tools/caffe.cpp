#ifdef WITH_PYTHON_LAYER
#include "boost/python.hpp"
namespace bp = boost::python;
#endif

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/signal_handler.h"

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::Solver;
using caffe::shared_ptr;
using caffe::string;
using caffe::Timer;
using caffe::vector;
using std::ostringstream;

DEFINE_string(gpu, "",
	"Optional; run in GPU mode on given device IDs separated by ','."
	"Use '-gpu all' to run on all available GPUs. The effective training "
	"batch size is multiplied by the number of devices.");
DEFINE_string(solver, "",
	"The solver definition protocol buffer text file.");
DEFINE_string(model, "",
	"The model definition protocol buffer text file.");
DEFINE_string(snapshot, "",
	"Optional; the snapshot solver state to resume training.");
DEFINE_string(weights, "",
	"Optional; the pretrained weights to initialize finetuning, "
	"separated by ','. Cannot be set simultaneously with snapshot.");
DEFINE_int32(iterations, 50,
	"The number of iterations to run.");
DEFINE_string(sigint_effect, "stop",
	"Optional; action to take when a SIGINT signal is received: "
	"snapshot, stop or none.");
DEFINE_string(sighup_effect, "snapshot",
	"Optional; action to take when a SIGHUP signal is received: "
	"snapshot, stop or none.");

// A simple registry for caffe commands.
typedef int(*BrewFunction)();
typedef std::map<caffe::string, BrewFunction> BrewMap;
BrewMap g_brew_map;

#define RegisterBrewFunction(func) \
namespace { \
class __Registerer_##func { \
 public: /* NOLINT */ \
  __Registerer_##func() { \
    g_brew_map[#func] = &func; \
  } \
}; \
__Registerer_##func g_registerer_##func; \
}

static BrewFunction GetBrewFunction(const caffe::string& name) {
	if (g_brew_map.count(name)) {
		return g_brew_map[name];
	}
	else {
		LOG(ERROR) << "Available caffe actions:";
		for (BrewMap::iterator it = g_brew_map.begin();
			it != g_brew_map.end(); ++it) {
			LOG(ERROR) << "\t" << it->first;
		}
		LOG(FATAL) << "Unknown action: " << name;
		return NULL;  // not reachable, just to suppress old compiler warnings.
	}
}

// Parse GPU ids or use all available devices
static void get_gpus(vector<int>* gpus) {
	if (FLAGS_gpu == "all") {
		int count = 0;
#ifndef CPU_ONLY
		CUDA_CHECK(cudaGetDeviceCount(&count));
#else
		NO_GPU;
#endif
		for (int i = 0; i < count; ++i) {
			gpus->push_back(i);
		}
	}
	else if (FLAGS_gpu.size()) {
		vector<string> strings;
		boost::split(strings, FLAGS_gpu, boost::is_any_of(","));
		for (int i = 0; i < strings.size(); ++i) {
			gpus->push_back(boost::lexical_cast<int>(strings[i]));
		}
	}
	else {
		CHECK_EQ(gpus->size(), 0);
	}
}

// caffe commands to call by
//     caffe <command> <args>
//
// To add a command, define a function "int command()" and register it with
// RegisterBrewFunction(action);

// Device Query: show diagnostic information for a GPU device.
int device_query() {
	LOG(INFO) << "Querying GPUs " << FLAGS_gpu;
	vector<int> gpus;
	get_gpus(&gpus);
	for (int i = 0; i < gpus.size(); ++i) {
		caffe::Caffe::SetDevice(gpus[i]);
		caffe::Caffe::DeviceQuery();
	}
	return 0;
}
RegisterBrewFunction(device_query);

// Load the weights from the specified caffemodel(s) into the train and
// test nets.
void CopyLayers(caffe::Solver<float>* solver, const std::string& model_list) {
	std::vector<std::string> model_names;
	boost::split(model_names, model_list, boost::is_any_of(","));
	for (int i = 0; i < model_names.size(); ++i) {
		LOG(INFO) << "Finetuning from " << model_names[i];
		solver->net()->CopyTrainedLayersFrom(model_names[i]);
		for (int j = 0; j < solver->test_nets().size(); ++j) {
			solver->test_nets()[j]->CopyTrainedLayersFrom(model_names[i]);
		}
	}
}

// Translate the signal effect the user specified on the command-line to the
// corresponding enumeration.
caffe::SolverAction::Enum GetRequestedAction(
	const std::string& flag_value) {
	if (flag_value == "stop") {
		return caffe::SolverAction::STOP;
	}
	if (flag_value == "snapshot") {
		return caffe::SolverAction::SNAPSHOT;
	}
	if (flag_value == "none") {
		return caffe::SolverAction::NONE;
	}
	LOG(FATAL) << "Invalid signal effect \"" << flag_value << "\" was specified";
	return caffe::SolverAction::NONE;
}

// Train / Finetune a model.
int train() {
	CHECK_GT(FLAGS_solver.size(), 0) << "Need a solver definition to train.";
	CHECK(!FLAGS_snapshot.size() || !FLAGS_weights.size())
		<< "Give a snapshot to resume training or weights to finetune "
		"but not both.";

	caffe::SolverParameter solver_param;
	caffe::ReadSolverParamsFromTextFileOrDie(FLAGS_solver, &solver_param);

	// If the gpus flag is not provided, allow the mode and device to be set
	// in the solver prototxt.
	if (FLAGS_gpu.size() == 0
		&& solver_param.solver_mode() == caffe::SolverParameter_SolverMode_GPU) {
		if (solver_param.has_device_id()) {
			FLAGS_gpu = "" +
				boost::lexical_cast<string>(solver_param.device_id());
		}
		else {  // Set default GPU if unspecified
			FLAGS_gpu = "" + boost::lexical_cast<string>(0);
		}
	}

	vector<int> gpus;
	get_gpus(&gpus);
	if (gpus.size() == 0) {
		LOG(INFO) << "Use CPU.";
		Caffe::set_mode(Caffe::CPU);
	}
	else {
		ostringstream s;
		for (int i = 0; i < gpus.size(); ++i) {
			s << (i ? ", " : "") << gpus[i];
		}
		LOG(INFO) << "Using GPUs " << s.str();
#ifndef CPU_ONLY
		cudaDeviceProp device_prop;
		for (int i = 0; i < gpus.size(); ++i) {
			cudaGetDeviceProperties(&device_prop, gpus[i]);
			LOG(INFO) << "GPU " << gpus[i] << ": " << device_prop.name;
		}
#endif
		solver_param.set_device_id(gpus[0]);
		Caffe::SetDevice(gpus[0]);
		Caffe::set_mode(Caffe::GPU);
		Caffe::set_solver_count(gpus.size());
	}

	caffe::SignalHandler signal_handler(
		GetRequestedAction(FLAGS_sigint_effect),
		GetRequestedAction(FLAGS_sighup_effect));

	shared_ptr<caffe::Solver<float> >
		solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));

	solver->SetActionFunction(signal_handler.GetActionFunction());

	if (FLAGS_snapshot.size()) {
		LOG(INFO) << "Resuming from " << FLAGS_snapshot;
		solver->Restore(FLAGS_snapshot.c_str());
	}
	else if (FLAGS_weights.size()) {
		CopyLayers(solver.get(), FLAGS_weights);
	}

	if (gpus.size() > 1) {
		caffe::P2PSync<float> sync(solver, NULL, solver->param());
		sync.Run(gpus);
	}
	else {
		LOG(INFO) << "Starting Optimization";
		solver->Solve();
	}
	LOG(INFO) << "Optimization Done.";
	return 0;
}
RegisterBrewFunction(train);


// Test: score a model.
int test() {
	CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";
	CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";

	// Set device id and mode
	vector<int> gpus;
	get_gpus(&gpus);
	if (gpus.size() != 0) {
		LOG(INFO) << "Use GPU with device ID " << gpus[0];
#ifndef CPU_ONLY
		cudaDeviceProp device_prop;
		cudaGetDeviceProperties(&device_prop, gpus[0]);
		LOG(INFO) << "GPU device name: " << device_prop.name;
#endif
		Caffe::SetDevice(gpus[0]);
		Caffe::set_mode(Caffe::GPU);
	}
	else {
		LOG(INFO) << "Use CPU.";
		Caffe::set_mode(Caffe::CPU);
	}
	// Instantiate the caffe net.
	Net<float> caffe_net(FLAGS_model, caffe::TEST);
	caffe_net.CopyTrainedLayersFrom(FLAGS_weights);
	LOG(INFO) << "Running for " << FLAGS_iterations << " iterations.";

	vector<int> test_score_output_id;
	vector<float> test_score;
	float loss = 0;
	for (int i = 0; i < FLAGS_iterations; ++i) {
		float iter_loss;
		const vector<Blob<float>*>& result =
			caffe_net.Forward(&iter_loss);
		loss += iter_loss;
		int idx = 0;
		for (int j = 0; j < result.size(); ++j) {
			const float* result_vec = result[j]->cpu_data();
			for (int k = 0; k < result[j]->count(); ++k, ++idx) {
				const float score = result_vec[k];
				if (i == 0) {
					test_score.push_back(score);
					test_score_output_id.push_back(j);
				}
				else {
					test_score[idx] += score;
				}
				const std::string& output_name = caffe_net.blob_names()[
					caffe_net.output_blob_indices()[j]];
				LOG(INFO) << "Batch " << i << ", " << output_name << " = " << score;
			}
		}
	}
	loss /= FLAGS_iterations;
	LOG(INFO) << "Loss: " << loss;
	for (int i = 0; i < test_score.size(); ++i) {
		const std::string& output_name = caffe_net.blob_names()[
			caffe_net.output_blob_indices()[test_score_output_id[i]]];
		const float loss_weight = caffe_net.blob_loss_weights()[
			caffe_net.output_blob_indices()[test_score_output_id[i]]];
		std::ostringstream loss_msg_stream;
		const float mean_score = test_score[i] / FLAGS_iterations;
		if (loss_weight) {
			loss_msg_stream << " (* " << loss_weight
				<< " = " << loss_weight * mean_score << " loss)";
		}
		LOG(INFO) << output_name << " = " << mean_score << loss_msg_stream.str();
	}

	return 0;
}
RegisterBrewFunction(test);


// Time: benchmark the execution time of a model.
int time() {
	CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to time.";

	// Set device id and mode
	vector<int> gpus;
	get_gpus(&gpus);
	if (gpus.size() != 0) {
		LOG(INFO) << "Use GPU with device ID " << gpus[0];
		Caffe::SetDevice(gpus[0]);
		Caffe::set_mode(Caffe::GPU);
	}
	else {
		LOG(INFO) << "Use CPU.";
		Caffe::set_mode(Caffe::CPU);
	}
	// Instantiate the caffe net.
	Net<float> caffe_net(FLAGS_model, caffe::TRAIN);

	// Do a clean forward and backward pass, so that memory allocation are done
	// and future iterations will be more stable.
	LOG(INFO) << "Performing Forward";
	// Note that for the speed benchmark, we will assume that the network does
	// not take any input blobs.
	float initial_loss;
	caffe_net.Forward(&initial_loss);
	LOG(INFO) << "Initial loss: " << initial_loss;
	LOG(INFO) << "Performing Backward";
	caffe_net.Backward();

	const vector<shared_ptr<Layer<float> > >& layers = caffe_net.layers();
	const vector<vector<Blob<float>*> >& bottom_vecs = caffe_net.bottom_vecs();
	const vector<vector<Blob<float>*> >& top_vecs = caffe_net.top_vecs();
	const vector<vector<bool> >& bottom_need_backward =
		caffe_net.bottom_need_backward();
	LOG(INFO) << "*** Benchmark begins ***";
	LOG(INFO) << "Testing for " << FLAGS_iterations << " iterations.";
	Timer total_timer;
	total_timer.Start();
	Timer forward_timer;
	Timer backward_timer;
	Timer timer;
	std::vector<double> forward_time_per_layer(layers.size(), 0.0);
	std::vector<double> backward_time_per_layer(layers.size(), 0.0);
	double forward_time = 0.0;
	double backward_time = 0.0;
	for (int j = 0; j < FLAGS_iterations; ++j) {
		Timer iter_timer;
		iter_timer.Start();
		forward_timer.Start();
		for (int i = 0; i < layers.size(); ++i) {
			timer.Start();
			layers[i]->Forward(bottom_vecs[i], top_vecs[i]);
			forward_time_per_layer[i] += timer.MicroSeconds();
		}
		forward_time += forward_timer.MicroSeconds();
		backward_timer.Start();
		for (int i = layers.size() - 1; i >= 0; --i) {
			timer.Start();
			layers[i]->Backward(top_vecs[i], bottom_need_backward[i],
				bottom_vecs[i]);
			backward_time_per_layer[i] += timer.MicroSeconds();
		}
		backward_time += backward_timer.MicroSeconds();
		LOG(INFO) << "Iteration: " << j + 1 << " forward-backward time: "
			<< iter_timer.MilliSeconds() << " ms.";
	}
	LOG(INFO) << "Average time per layer: ";
	for (int i = 0; i < layers.size(); ++i) {
		const caffe::string& layername = layers[i]->layer_param().name();
		LOG(INFO) << std::setfill(' ') << std::setw(10) << layername <<
			"\tforward: " << forward_time_per_layer[i] / 1000 /
			FLAGS_iterations << " ms.";
		LOG(INFO) << std::setfill(' ') << std::setw(10) << layername <<
			"\tbackward: " << backward_time_per_layer[i] / 1000 /
			FLAGS_iterations << " ms.";
	}
	total_timer.Stop();
	LOG(INFO) << "Average Forward pass: " << forward_time / 1000 /
		FLAGS_iterations << " ms.";
	LOG(INFO) << "Average Backward pass: " << backward_time / 1000 /
		FLAGS_iterations << " ms.";
	LOG(INFO) << "Average Forward-Backward: " << total_timer.MilliSeconds() /
		FLAGS_iterations << " ms.";
	LOG(INFO) << "Total Time: " << total_timer.MilliSeconds() << " ms.";
	LOG(INFO) << "*** Benchmark ends ***";
	return 0;
}
RegisterBrewFunction(time);

// Getfeature: calculate caffe modal and output the result. (user)
int getfeature() {
	CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";
	CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";

	// Set device id and mode
	vector<int> gpus;
	get_gpus(&gpus);
	if (gpus.size() != 0) {
		LOG(INFO) << "Use GPU with device ID " << gpus[0];
#ifndef CPU_ONLY
		cudaDeviceProp device_prop;
		cudaGetDeviceProperties(&device_prop, gpus[0]);
		LOG(INFO) << "GPU device name: " << device_prop.name;
#endif
		Caffe::SetDevice(gpus[0]);
		Caffe::set_mode(Caffe::GPU);
	}
	else {
		LOG(INFO) << "Use CPU.";
		Caffe::set_mode(Caffe::CPU);
	}
	// Instantiate the caffe net.
	Net<float> caffe_net(FLAGS_model, caffe::TEST);
	caffe_net.CopyTrainedLayersFrom(FLAGS_weights);

	CHECK_EQ(caffe_net.num_outputs(), 1) << "features must be in 1 output blob.";
	int layer_id = 0;
	while (layer_id < caffe_net.layers().size())
	{
		std::string layer_type(caffe_net.layers()[layer_id].get()->type());
		if (layer_type.compare("ImageData") == 0) break;
		layer_id++;
	}
	CHECK_LT(layer_id, caffe_net.layers().size()) << "Only for ImageData input.";
	LOG(INFO) << "image_data_layer: " << layer_id;

	caffe::LayerParameter layer_param = caffe_net.layers()[layer_id].get()->layer_param();
	std::string sourcename = layer_param.image_data_param().source();
	LOG(INFO) << "Read from: " << sourcename;
	CHECK_EQ(layer_param.image_data_param().batch_size(), 1) << "Set batch_size to 1 for counting iterations.";
	CHECK_EQ(layer_param.image_data_param().shuffle(), false) << "Data must be not shuffled.";
	std::ifstream prereadfile(sourcename);
	int count = 0;
	std::string fname;
	int label;
	while (prereadfile >> fname >> label) count++;
	prereadfile.close();
	FLAGS_iterations = count;
	LOG(INFO) << "Running for " << FLAGS_iterations << " iterations.";

	int idx = sourcename.find_last_of('.');
	std::string outputname(sourcename.substr(0, idx) + ".caffedata");
	LOG(INFO) << "Write to: " << outputname;
	std::ofstream outputfile;

	//vector<Blob<float>* > bottom_vec;
	vector<int> test_score_output_id;
	vector<float> test_score;
	float loss = 0;
	int feature_num = caffe_net.output_blobs()[0]->count();  //batch_size == 1
	LOG(INFO) << "feature_num: " << feature_num;
	for (int i = 0; i < FLAGS_iterations; ++i) {
		float iter_loss;
		const vector<Blob<float>*>& result =
			caffe_net.Forward(&iter_loss);

		if (i == 0)
		{
			outputfile.open(outputname, std::ios::binary);
			outputfile.write((char*)&feature_num, sizeof(int));
		}
		const float* data = result[0]->cpu_data();
		outputfile.write((char*)data, feature_num * sizeof(float));

		if (i % 100 == 0)
		{
			LOG(INFO) << "iteration: " << i;
		}
	}
	if (outputfile.is_open())
		outputfile.close();
	return 0;
}
RegisterBrewFunction(getfeature);

// extractweights: extract weights data from .caffemodel file and write to .weighttxt file and .weightbin file. (custom)
int extractweights() {
	CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to extract.";

	std::string sourcename(FLAGS_weights);
	int idx = FLAGS_weights.find_last_of('.');
	std::string outpathname(FLAGS_weights.substr(0, idx));
	std::ofstream fouttxt(outpathname + ".weighttxt");
	CHECK(fouttxt) << "Can not write to " << outpathname + ".weighttxt";
	std::ofstream foutbin;
	foutbin.open(outpathname + ".weightbin", std::ios::binary);
	CHECK(foutbin) << "Can not write to " << outpathname + ".weightbin";
	LOG(INFO) << "Write to " << outpathname + ".weighttxt";
	LOG(INFO) << "Write to " << outpathname + ".weightbin";

	//Read paramfile
	caffe::NetParameter param;
	caffe::ReadProtoFromBinaryFileOrDie(sourcename, &param);
	int num_source_layers = param.layer_size();
	for (int i = 0; i < num_source_layers; ++i) {
		const caffe::LayerParameter& source_layer = param.layer(i);
		for (int j = 0; j < source_layer.blobs_size(); j++)
		{
			int shape[4];
			for (int k = 0; k < 4; k++)
			{
				if (k < source_layer.blobs(j).shape().dim_size())
					shape[k] = source_layer.blobs(j).shape().dim(k);
				else
					shape[k] = 1;
			}
			int count = shape[0] * shape[1] * shape[2] * shape[3];
			//CHECK_EQ(source_layer.blobs(j).data_size(), count) << "Data size not equal at layer " << i;

			fouttxt << "//" << shape[0] << "," << shape[1] << "," << shape[2] << "," << shape[3] << std::endl;
			foutbin.write((const char *)shape, sizeof(int) * 4);
			LOG(INFO) << "Layer " << param.layer(i).name() << " blob " << j << ": "
				<< shape[0] << "," << shape[1] << "," << shape[2] << "," << shape[3];

			fouttxt << "float ";
			//switch (j)
			//{
			//case 0:
			//	fouttxt << "w_";
			//	break;
			//case 1:
			//	fouttxt << "b_";
			//	break;
			//default:
			//	fouttxt << "param" << j << "_";
			//	break;
			//}
			fouttxt << source_layer.name() << "_b" << j << "[] = {";
			for (int k = 0; k < source_layer.blobs(j).data_size(); k++)
			{
				float curdata = source_layer.blobs(j).data(k);
				if (k > 0)
				{
					fouttxt << ",";
					if (k % 1024 == 0)
						fouttxt << std::endl;
				}

				fouttxt << curdata;
				foutbin.write((const char *)&curdata, sizeof(float));
			}
			fouttxt << "};" << std::endl;
		}
	}
	return 0;
}
RegisterBrewFunction(extractweights);

// absorb: absorb batch-normalization layer to convolution layer or inner-product layer. (custom)
int absorb() {
	CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to extract.";

	std::string sourcename(FLAGS_weights);
	int idx = FLAGS_weights.find_last_of('.');
	std::string outpathname(FLAGS_weights.substr(0, idx));
	std::string outmodel(outpathname + "_n.caffemodel");

	//Read paramfile
	caffe::NetParameter param;
	caffe::ReadProtoFromBinaryFileOrDie(sourcename, &param);

	Blob<double> nscale;
	Blob<double> nbias;
	vector<int> shape(1, 0);
	nscale.Reshape(shape);
	nbias.Reshape(shape);
	//Blob<float> tempscale;
	//Blob<float> tempbias;
	//Blob<float> tempS;
	//Blob<float> temp1;
	int src_layer_id = param.layer_size() - 1;
	for (; src_layer_id >= 0; src_layer_id--) {
		const caffe::LayerParameter& source_layer = param.layer(src_layer_id);
		Blob<double> tempscale;
		Blob<double> tempbias;
		if ((source_layer.type() == "Scale") || (source_layer.type() == "BatchNorm"))
		{
			LOG(INFO) << "Calculate datas from " << source_layer.type() << " Layer: " << source_layer.name();
			if (source_layer.type() == "Scale")
			{
				tempscale.FromProto(source_layer.blobs(0));
				if (source_layer.scale_param().bias_term())
				{
					tempbias.FromProto(source_layer.blobs(1));
				}
				else
				{
					tempbias.ReshapeLike(tempscale);
					caffe::caffe_set(tempbias.count(), (double)0, tempbias.mutable_cpu_data());
				}
			}
			else if (source_layer.type() == "BatchNorm")
			{
				Blob<double> tempS;
				double eps = source_layer.batch_norm_param().eps();
				tempbias.FromProto(source_layer.blobs(0));
				tempscale.FromProto(source_layer.blobs(1));
				double scale_factor = source_layer.blobs(2).data(0) == 0 ? 0 : 1 / source_layer.blobs(2).data(0);
				caffe::caffe_scal(tempbias.count(), -scale_factor, tempbias.mutable_cpu_data());
				caffe::caffe_scal(tempscale.count(), scale_factor, tempscale.mutable_cpu_data());
				caffe::caffe_add_scalar(tempscale.count(), eps, tempscale.mutable_cpu_data());
				caffe::caffe_powx(tempscale.count(), tempscale.cpu_data(), (double)0.5, tempscale.mutable_cpu_data());

				caffe::caffe_div(tempbias.count(), tempbias.cpu_data(), tempscale.cpu_data(), tempbias.mutable_cpu_data());
				tempS.ReshapeLike(tempscale);
				caffe::caffe_set(tempS.count(), (double)1, tempS.mutable_cpu_data());
				caffe::caffe_div(tempS.count(), tempS.cpu_data(), tempscale.cpu_data(), tempscale.mutable_cpu_data());
			}
			if (nscale.count() == 0 && nbias.count() == 0)
			{
				CHECK_EQ(source_layer.bottom_size(), 1);
				CHECK_EQ(source_layer.top_size(), 1);
				CHECK_LT(src_layer_id + 1, param.layer_size());
				for (int layer_id = src_layer_id + 1; layer_id < param.layer_size(); layer_id++)
				{
					for (int i = 0; i < param.layer(layer_id).bottom_size(); i++)
						if (param.layer(layer_id).bottom(i) == source_layer.top(0))
						{
							param.mutable_layer(layer_id)->set_bottom(i, source_layer.bottom(0));
						}
				}
				param.mutable_layer(src_layer_id + 1)->set_bottom(0, source_layer.bottom(0));
				nscale.CopyFrom(tempscale, false, true);
				nbias.CopyFrom(tempbias, false, true);
			}
			else
			{
				caffe::caffe_mul(tempbias.count(), tempbias.cpu_data(), nscale.cpu_data(), tempbias.mutable_cpu_data());
				caffe::caffe_add(nbias.count(), nbias.cpu_data(), tempbias.cpu_data(), nbias.mutable_cpu_data());
				caffe::caffe_mul(nscale.count(), nscale.cpu_data(), tempscale.cpu_data(), nscale.mutable_cpu_data());
			}
		}
		else if (((source_layer.type() == "Convolution") || (source_layer.type() == "Deconvolution") || (source_layer.type() == "InnerProduct")) && (nscale.count() > 0 || nbias.count() > 0))
		{
			LOG(INFO) << "Calculate datas to " << source_layer.type() << " Layer: " << source_layer.name();
			Blob<double> cweight;
			Blob<double> cbias;			
			cweight.FromProto(source_layer.blobs(0));
			if (source_layer.blobs_size() > 1)
			{
				cbias.FromProto(source_layer.blobs(1));
				caffe::caffe_mul(cbias.count(), cbias.cpu_data(), nscale.cpu_data(), cbias.mutable_cpu_data());
				caffe::caffe_add(cbias.count(), nbias.cpu_data(), cbias.cpu_data(), cbias.mutable_cpu_data());
			}
			else
			{
				cbias.CopyFrom(nbias, false, true);
			}
			int dim = cweight.count(1);
			double* cweight_data = cweight.mutable_cpu_data();
			const double* nscale_data = nscale.cpu_data();
			for (int n = 0; n < cweight.shape(0); n++)
				for (int d = 0; d < dim; d++)
				{
					cweight_data[n * dim + d] *= nscale_data[n];
				}

			caffe::LayerParameter* templayer = param.mutable_layer(src_layer_id);
			templayer->clear_blobs();

			Blob<float> oweight(cweight.shape());
			Blob<float> obias(cbias.shape());
			for (int i = 0; i < cweight.count(); i++) oweight.mutable_cpu_data()[i] = float(cweight.cpu_data()[i]);
			for (int i = 0; i < cbias.count(); i++) obias.mutable_cpu_data()[i] = float(cbias.cpu_data()[i]);

			oweight.ToProto(templayer->add_blobs(), false);
			obias.ToProto(templayer->add_blobs(), false);
			if (templayer->has_convolution_param())
				templayer->mutable_convolution_param()->set_bias_term(true);
			if (templayer->has_inner_product_param())
				templayer->mutable_inner_product_param()->set_bias_term(true);

			vector<int> shape(1, 0);
			nscale.Reshape(shape);
			nbias.Reshape(shape);
		}
	}

	caffe::NetParameter newparam = param;
	newparam.clear_layer();
	for (int src_layer_id = 0; src_layer_id < param.layer_size(); src_layer_id++)
	{
		const caffe::LayerParameter& source_layer = param.layer(src_layer_id);
		if (!((source_layer.type() == "Scale") || (source_layer.type() == "BatchNorm")))
		{
			LOG(INFO) << "Rewrite " << source_layer.type() << " Layer: " << source_layer.name();
			newparam.add_layer()->CopyFrom(source_layer);
		}
	}
	LOG(INFO) << "Save to " << outmodel;
	caffe::WriteProtoToBinaryFile(newparam, outmodel);
	return 0;
}
RegisterBrewFunction(absorb);

int main(int argc, char** argv) {
	// Print output to stderr (while still logging).
	FLAGS_alsologtostderr = 1;
	FLAGS_log_dir = "./LOG/";
	// Set version
	gflags::SetVersionString(AS_STRING(CAFFE_VERSION));
	// Usage message.
	gflags::SetUsageMessage("command line brew\n"
		"usage: caffe <command> <args>\n\n"
		"commands:\n"
		"  train           train or finetune a model\n"
		"  test            score a model\n"
		"  device_query    show GPU diagnostic information\n"
		"  time            benchmark model execution time");
	// Run tool or show usage.
	caffe::GlobalInit(&argc, &argv);
	if (argc == 2) {
#ifdef WITH_PYTHON_LAYER
		try {
#endif
			return GetBrewFunction(caffe::string(argv[1]))();
#ifdef WITH_PYTHON_LAYER
		}
		catch (bp::error_already_set) {
			PyErr_Print();
			return 1;
		}
#endif
	}
	else {
		gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/caffe");
	}
}
