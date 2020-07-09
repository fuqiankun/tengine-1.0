#include <string>
#include <sys/time.h>
#include "mtcnn.hpp"
#include "mtcnn_utils.hpp"
#include "common.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
 
int main(int argc, char** argv)
{

    const std::string model_dir =  "./models";
    const std::string proto_name = "./models/reg.proto";
    const std::string model_name = "./models/reg.model";
    if(init_tengine() < 0)
    {
        std::cout << " init tengine failed\n";
        return 1;
    }
    if(request_tengine_version("0.9") < 0)
    {
        std::cout << " request tengine version failed\n";
        return 1;
    }

    std::vector<std::vector<float>> id_pool;

    int min_size = 40;

    float conf_p = 0.6;
    float conf_r = 0.7;
    float conf_o = 0.8;

    float nms_p = 0.5;
    float nms_r = 0.7;
    float nms_o = 0.7;
    
    int img_h = 128;
    int img_w = 128;
    int img_size = img_h * img_w;
    float *input_data = (float*)malloc(sizeof(float) * img_size);

    mtcnn* det = new mtcnn(min_size, conf_p, conf_r, conf_o, nms_p, nms_r, nms_o);
    det->load_3model(model_dir);
    
    graph_t graph = create_graph(nullptr, "caffe", proto_name.c_str(), model_name.c_str());

    if(graph == nullptr)
    {
        std::cout << "create reg failed\n";
        return 1;
    }
    std::cout << "create reg done!\n";
    
    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
    int dims[] = {1, 1, img_h, img_w};
    set_tensor_shape(input_tensor, dims, 4);
    if(set_tensor_buffer(input_tensor, input_data, img_size * 4) < 0)
    {
        std::printf("set buffer for input tensor failed\n");
        return -1;
    }
    
    int ret = -1;
    ret = prerun_graph(graph);
    if(ret != 0)
    {
        std::cout << "Prerun graph failed, errno: " << get_tengine_errno() << "\n";
        return 1;
    }


    cv:: VideoCapture cap(0);
    if(!cap.isOpened())
        std::cout << "fail to open camera!" << std::endl;
    
    cv::Mat image;
    while(cap.read(image))
    {
	std::vector<face_box> face_info;    
    	det->detect(image, face_info);
	for(unsigned int i = 0; i < face_info.size(); i++)
    	{
            face_box& box = face_info[i];
            std::printf("BOX:( %g , %g ),( %g , %g )\n", box.x0, box.y0, box.x1, box.y1);
            cv::rectangle(image, cv::Point(box.x0, box.y0), cv::Point(box.x1, box.y1), cv::Scalar(0, 255, 0), 1);
         
    	}
    	if(face_info.size() > 0){
	    face_box& box = face_info[0];
            if(!(box.x0 < 0 || box.x1 > 639 || box.y0 < 0 || box.y1 > 479))
	    {
	    	cv::Rect rect(box.x0, box.y0, box.x1 - box.x0, box.y1 - box.y0);
            	cv::Mat crop_face = image(rect);
            	cv::cvtColor(crop_face, crop_face, CV_RGB2GRAY);
            	cv::resize(crop_face, crop_face, cv::Size(128,128), 0, 0, CV_INTER_LINEAR);

	    	for(int i = 0;i < 128;i++){
			for(int j = 0;j < 128;j++){
				input_data[i * 128 + j] = crop_face.ptr<uchar>(i)[j] / 255.0;
			}
	    	}
	    	ret = run_graph(graph, 1);//compute 256 dimension embedding
            	if(ret != 0)
            	{
                	std::cout << "Run graph failed, errno: " << get_tengine_errno() << "\n";
            		return 1;
            	}
	    	int size1 = 256;
    	    	tensor_t mytensor1 = get_graph_tensor(graph, "eltwise_fc1");
    	    	float* face_embedding = ( float* )get_tensor_buffer(mytensor1);// face embedding float[256]
	    	std::cout << "reg" << std::endl;
	    }        
	}    	
    }

    delete det;
    free(input_data);
    release_tengine();

    return 0;
}
