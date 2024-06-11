#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#include <cuda_provider_factory.h>
#include <onnxruntime_cxx_api.h>

using namespace cv;
using namespace std;
using namespace Ort;


class LYTNet
{
public:
	LYTNet(string model_path);
	Mat detect(Mat srcimg);
private:
	vector<float> input_image_;
	int inpWidth;
	int inpHeight;
	int outWidth;
	int outHeight;
	const float score_th = 0;

	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "Low-Light Image Enhancement");
	Ort::Session *ort_session = nullptr;
	SessionOptions sessionOptions = SessionOptions();
	vector<char*> input_names;
	vector<char*> output_names;
	vector<vector<int64_t>> input_node_dims; // >=1 outputs
	vector<vector<int64_t>> output_node_dims; // >=1 outputs
};

LYTNet::LYTNet(string model_path)
{
	/// OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);   ///cuda

	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	/*std::wstring widestr = std::wstring(model_path.begin(), model_path.end()); ////windows
	ort_session = new Session(env, widestr.c_str(), sessionOptions); ////windows*/
	ort_session = new Session(env, model_path.c_str(), sessionOptions); ////linux

	size_t numInputNodes = ort_session->GetInputCount();
	size_t numOutputNodes = ort_session->GetOutputCount();
	AllocatorWithDefaultOptions allocator;
	for (int i = 0; i < numInputNodes; i++)
	{
		input_names.push_back(ort_session->GetInputName(i, allocator));
		Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		auto input_dims = input_tensor_info.GetShape();
		input_node_dims.push_back(input_dims);
	}
	for (int i = 0; i < numOutputNodes; i++)
	{
		output_names.push_back(ort_session->GetOutputName(i, allocator));
		Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		auto output_dims = output_tensor_info.GetShape();
		output_node_dims.push_back(output_dims);
	}
	this->inpHeight = input_node_dims[0][1];
	this->inpWidth = input_node_dims[0][2];
	this->outHeight = output_node_dims[0][1];
	this->outWidth = output_node_dims[0][2];
}


/***************** Mat转vector **********************/
template<typename _Tp>
vector<_Tp> convertMat2Vector(const Mat &mat)
{
	return (vector<_Tp>)(mat.reshape(1, 1));//通道数不变，按行转为一行
}
 
/****************** vector转Mat *********************/
template<typename _Tp>
cv::Mat convertVector2Mat(vector<_Tp> v, int channels, int rows)
{
	cv::Mat mat = cv::Mat(v).clone();//将vector变成单列的mat，这里需要clone(),因为这里的赋值操作是浅拷贝
	cv::Mat dest = mat.reshape(channels, rows);
	return dest;
}


Mat LYTNet::detect(Mat srcimg)
{
	Mat dstimg;
	resize(srcimg, dstimg, Size(this->inpWidth, this->inpHeight));
	dstimg.convertTo(dstimg, CV_32FC3, 1 / 127.5, -1.0);
	this->input_image_ = (vector<float>)(dstimg.reshape(1,1));

	// const size_t area = this->inpWidth * this->inpHeight * 3;
	// this->input_image_.resize(area);
	// memcpy(this->input_image_.data(), (float*)dstimg.data, area*sizeof(float));
	
	array<int64_t, 4> input_shape_{ 1, this->inpHeight, this->inpWidth, 3 };

	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());

	vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr }, &input_names[0], &input_tensor_, 1, output_names.data(), output_names.size());   // ��ʼ����
	
	float* pred = ort_outputs[0].GetTensorMutableData<float>();
	Mat output_image(outHeight, outWidth, CV_32FC3, pred);
	output_image = (output_image + 1.0 ) * 127.5;
	output_image.convertTo(output_image, CV_8UC3);
	resize(output_image, output_image, Size(srcimg.cols, srcimg.rows));
	return output_image;
}

int main()
{
	LYTNet mynet("weights/lyt_net_lolv2_real_320x240.onnx");
	string imgpath = "testimgs/1_1.JPG";
	Mat srcimg = imread(imgpath);
	
	Mat dstimg = mynet.detect(srcimg);
	
	namedWindow("srcimg", WINDOW_NORMAL);
	imshow("srcimg", srcimg);
	namedWindow("dstimg", WINDOW_NORMAL);
	imshow("dstimg", dstimg);
	waitKey(0);
	destroyAllWindows();
}