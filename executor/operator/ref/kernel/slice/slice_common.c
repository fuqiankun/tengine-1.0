/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * License); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * Copyright (c) 2019, Open AI Lab
 * Author: ruizhang@openailab.com
 */
  
static int caffe_run(const int8_t* in_data,int8_t** out_data,int element_size,const struct slice_param *param)
{
    // get the slice param
    int slice_axis = param->axis;
    int num_slices = 1;
    int slice_size = 1; 
    const int8_t * input = in_data;
    const int *in_dim = param->in_shape;
   
    for(int i = 0; i < slice_axis; i++)
    {
        num_slices = num_slices * in_dim[i];
    }
    for(int i = slice_axis + 1; i < param->dim_num; i++)
    {
        slice_size = slice_size * in_dim[i];
    }
    int in_slice = in_dim[slice_axis];
    int slice_index = 0;
    int out_num = param->out_num;
    for(int i = 0; i < out_num; i++)
    {
          int8_t* output = out_data[i];
          int out_slice = param->output_shape[i].dims[slice_axis];        
          for(int n = 0; n < num_slices; n++)
          {
               int in_offset = (n * in_slice + slice_index) * slice_size * element_size;
               int out_offset  = n * out_slice * slice_size * element_size;
               memcpy(output+out_offset,input+in_offset,slice_size * out_slice * element_size);
          }
          slice_index += out_slice;       
    }
    return 0;
    
}
static int tf_run(const int8_t* in_data,int8_t** out_data,int element_size,const struct slice_param *param)
{
    const int8_t* input = in_data;
    int8_t* output = out_data[0];
   
    const int *begins = param->output_shape[0].begins;
    const int *sizes = param->output_shape[0].sizes;
    int real_dim = param->dim_num;
    const int* in_dim_new = param->in_shape;
    int in_dim_0 = in_dim_new[0];
    int in_dim_1 = in_dim_new[1];
    int in_dim_2 = in_dim_new[2];
    int in_dim_3 = in_dim_new[3];

    int start_dim_0 = (4 - real_dim) > 0 ? 0 : begins[0];
    int stop_dim_0 = ((4 - real_dim) > 0 || sizes[0] == -1)
                 ? in_dim_0 - start_dim_0
                 : start_dim_0 + sizes[0];
    int start_dim_1 = (3 - real_dim) > 0 ? 0 : begins[1];
    int stop_dim_1 = ((3 - real_dim) > 0 || sizes[1] == -1)
                 ? in_dim_1 - start_dim_1
                 : start_dim_1 + sizes[1];             
    int start_dim_2 = (2 - real_dim) > 0 ? 0 : begins[2];
    int stop_dim_2 = ((2 - real_dim) > 0 || sizes[2] == -1)
                 ? in_dim_2 - start_dim_2
                 : start_dim_2 + sizes[2];             
    int start_dim_3 = (1 - real_dim) > 0 ? 0 : begins[3];
    int stop_dim_3 = ((1 - real_dim) > 0 || sizes[3] == -1)
                 ? in_dim_3 - start_dim_3
                 : start_dim_3 + sizes[3];             
    
    for(int n = start_dim_0; n < stop_dim_0;++n)
    {
        for(int i = start_dim_1; i < stop_dim_1; ++i)
        {
            for(int j = start_dim_2; j < stop_dim_2; ++j)
            {
                int len = stop_dim_3 - start_dim_3;
                int input_off = n * in_dim_1 * in_dim_2 * in_dim_3 + 
                                i * in_dim_2 * in_dim_3 + 
                                j * in_dim_3 + start_dim_3;
                memcpy(output,input + input_off,len * element_size);
                output += len * element_size;
            }
        }
    }
    return 0;
}
static int ref_slice_common(const int8_t* in_data,int8_t** out_data,int element_size,const struct slice_param *param)
{
    if(param->iscaffe)
        return caffe_run(in_data,out_data,element_size,param);
    else
        return tf_run(in_data,out_data,element_size,param);
     
}
