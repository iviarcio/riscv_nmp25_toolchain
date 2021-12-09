# Copyright 2018 LG Electoronics All rights reserved.

import os
import sys
import argparse
import numpy as np
import copy
sys.path.append(os.path.join(os.path.dirname(__file__), './flatbuffers/python/'))
import flatbuffers
import tflite.Model
import tflite.Buffer
import tflite.Operator
import tflite.SubGraph
import tflite.TensorType
import tflite.Tensor
import tflite.OperatorCode
import tflite.LneExecOptions
import tflite.SoftmaxOptions
import tflite.BuiltinOptions
import tflite.BuiltinOperator
import tflite.mean_vec

def find_layernumber(layer, name):
    for i in range(len(layer)):
        if layer[i]['name'] == name:
            break;
    return i

def find_layernumber_blob(layer, name):
    for i in range(len(layer)):
        if layer[i]['oblob'] == name:
            break;
    return i


def main(firmware, parameter, runscript, inout_info, mean, scale, input_type, output_type, idata_format, odata_format, post_processing, version):
# data for .lne schema
        lne_version = 3
        lne_hw_version = version
        fsize = 0
        psize = 0
        bsize = 0
        input_shape = []
        output_shape = []
        input_num = 0
        output_num = 0
        input_n=[]
        output_n=[]
        inputq = []
        outputq = []
        mean_b = float(mean[0])
        mean_g = float(mean[1])
        mean_r = float(mean[2])
        num_layer = 0
        LAYERS = []
        attr = {'name': '','oblob': '','size' : 0, 'w' : 0, 'h' : 0, "c" : 0, "n" : 0, "qmode" : "", "outq" : 0}
        with open(firmware, "rb") as f:
            firmware_buf = f.read()
            fsize = f.tell()

        with open(parameter, "rb") as w:
            parameter_buf = w.read()
            psize = w.tell()

    # bsize, inputq, outputq
        lines = open(runscript).readlines()
        for line in lines:
            str_temp = "".join(line.strip(' \t\n\r'))
            temp = str_temp.split("=")
            if line[0] == "L":
                ts = len(temp[0])
                if temp[0][ts - len("NAME"):] == "NAME":
                    attr['name'] = temp[1]
                if temp[0][ts - len("OUT_BLOB"):] == "OUT_BLOB":
                    attr['oblob'] = temp[1]
                if temp[0][ts - len("SIZE"):] == "SIZE":
                    attr['size'] = int(temp[1])
                if temp[0][ts - len("_W"):] == "_W":
                    attr['w'] = int(temp[1])
                if temp[0][ts - len("_H"):] == "_H":
                    attr['h'] = int(temp[1])
                if temp[0][ts - len("_CH"):] == "_CH":
                    attr['c'] = int(temp[1])
                if temp[0][ts - len("_NUM"):] == "_NUM":
                    attr['n'] = int(temp[1])
                if temp[0][ts - len("QMODE"):] == "QMODE":
                    attr['qmode'] = temp[1]
                    num_layer=num_layer + 1
                if temp[0][ts - len("Q"):] == "Q":
                    attr['outq'] = int(temp[1])
                    attr_temp = copy.deepcopy(attr)
                    LAYERS.append(attr_temp)

        # FXP16/FXP8
        q_mode = LAYERS[0]['qmode']
        if inout_info != "None":
       	    lines2 = open(inout_info).readlines()
            for line in lines2:
                str_temp = "".join(line.strip(' \t\n\r'))
                temp = str_temp.split("=")
                if temp[0] == "INPUT_NAME":
                    input_n.append(temp[1])
                elif temp[0] == "OUTPUT_NAME":
                    output_n.append(temp[1])
        else:
            layer_num = 0
            input_n.append(LAYERS[layer_num]['name'])
            print(LAYERS[layer_num]['name'])
            layer_num = len(LAYERS) - 1
            output_n.append(LAYERS[layer_num]['name'])
            print(LAYERS[layer_num]['name'])

        input_num = len(input_n)
        for i in range(input_num):
            temp = input_n[i].split(",")
            print(temp[0])
            layer_number = find_layernumber(LAYERS, temp[0])
            print(layer_number)
            inputq.append(LAYERS[layer_number]['outq'])
            temp_ch = int(LAYERS[layer_number]['c'])
            if len(temp) > 1: # for concat layer
                for j in range(len(temp)-1):
                    layer_number = find_layernumber(LAYERS, temp[j+1])
                    temp_ch += int(LAYERS[layer_number]['c'])

            if idata_format == "NHWC":
                input_shape.append([LAYERS[layer_number]['n'], LAYERS[layer_number]['h'], LAYERS[layer_number]['w'], temp_ch])
            elif idata_format == "NCHW":
                input_shape.append([LAYERS[layer_number]['n'], temp_ch, LAYERS[layer_number]['h'], LAYERS[layer_number]['w']])
            else:
                print("not support data format: ", idata_format)
                sys.exit(1);

        output_num = len(output_n)

        for i in range(output_num):
            temp = output_n[i].split(",")
            #layer_number = find_layernumber(LAYERS, temp[0])
            layer_number = find_layernumber_blob(LAYERS, temp[0])
            print(temp[0], ": ", layer_number)
            outputq.append(LAYERS[layer_number]['outq'])
            temp_ch = int(LAYERS[layer_number]['c'])
            if len(temp) > 1: # for concat layer
                for j in range(len(temp)-1):
                    layer_number = find_layernumber(LAYERS, temp[j+1])
                    temp_ch += int(LAYERS[layer_number]['c'])

            if odata_format == "NHWC":
                output_shape.append([LAYERS[layer_number]['n'], LAYERS[layer_number]['h'], LAYERS[layer_number]['w'], temp_ch])
            elif odata_format == "NCHW":
                output_shape.append([LAYERS[layer_number]['n'], temp_ch, LAYERS[layer_number]['h'], LAYERS[layer_number]['w']])
            else:
                print("not support data format: ", odata_format)
                sys.exit(1);

        for i in range(0, num_layer):
            bsize = bsize + (int((LAYERS[i]['size'] + 127) / 128)) * 128

        if fsize >= 1024*1024:
            print("firmware size is bigger than 1MB ", firmware, ",firmware size: ", fsize)
            print("DO NOT USE THIS FIRMWARE !!, Report to SIC(dq1-dev@lge.com)")
            print("DO NOT USE THIS FIRMWARE !!, Report to SIC(dq1-dev@lge.com)")
            print("DO NOT USE THIS FIRMWARE !!, Report to SIC(dq1-dev@lge.com)")
            print("DO NOT USE THIS FIRMWARE !!, Report to SIC(dq1-dev@lge.com)")
            print("DO NOT USE THIS FIRMWARE !!, Report to SIC(dq1-dev@lge.com)")
            sys.exit(1);

        if fsize + psize + bsize >= 240*1024*1024:
            print("Total file size exceeds 240MB, needs to do something")
            print("DO NOT USE THIS FIRMWARE !!, Report to SIC(dq1-dev@lge.com)")
            print("DO NOT USE THIS FIRMWARE !!, Report to SIC(dq1-dev@lge.com)")
            print("DO NOT USE THIS FIRMWARE !!, Report to SIC(dq1-dev@lge.com)")
            print("DO NOT USE THIS FIRMWARE !!, Report to SIC(dq1-dev@lge.com)")
            print("DO NOT USE THIS FIRMWARE !!, Report to SIC(dq1-dev@lge.com)")

        outputq_multi = 0
        inputq_multi = 0
        print("-------------------------------------------------")
        print("LNE Model Information")
        print("-------------------------------------------------")
        print("LNE HW Information")
        print("LNE HW version:", lne_hw_version)
        print("LNE Quantization mode: ", q_mode)
        print("-------------------------------------------------")
        print("LNE Network Information")
        print("firmware: ", firmware, ",firmware size: ", fsize)
        print("parameter: ", parameter, ",parameter size: ", psize)
        print("The number of layers: ", num_layer)
        print("-------------------------------------------------")
        print("Input Info")
        print("data format: ", idata_format)
        for i in range(input_num):
            print("number: ", i)
            print("shape: ", input_shape[i])
            print("q value: ", inputq[i])
            print("type: ", input_type)
            print("mean value(bgr order) ", mean_b, mean_g, mean_r)
            print("scale: ", scale)
            inputq_multi |= inputq[i] << (i * 4)
        print("-------------------------------------------------")
        print("Output Info")
        print("data format: ", odata_format)
        for i in range(output_num):
            print("number: ", i)
            print("shape: ", output_shape[i])
            print("q value: ", outputq[i])
            print("type: ", output_type)
            outputq_multi |= outputq[i] << (i * 4)


        #Experimental feature
        isSoftmax = False
        if args.w == "softmax":
            print("-------------------------------------------------")
            if output_num != 1:
                print("Not support softmax for multi output")
            elif odata_format != "NHWC":
                print("Not support softmax for NCHW output format")
            else:
                isSoftmax = True
                print("post processing: softmax")
                print("softmax output shape: [1, {}]".format(output_shape[0][3]))
                print("softmax output type: FLOAT32")

        builder = flatbuffers.Builder(8102)
        lne_desc = builder.CreateString('Flatbuffer for lne')
        sub_desc = builder.CreateString('Model SubGraph0')

        s_qmode = builder.CreateString(q_mode)
        s_idata_format = builder.CreateString(idata_format)
        s_odata_format = builder.CreateString(odata_format)

        #Tensor name
        firm_name = builder.CreateString('firmware')
        param_name = builder.CreateString('parameter')

        input_name = []
        for i in range(input_num):
            input_name.append(builder.CreateString('input'+str(i)))

        output_name = []
        for i in range(output_num):
            output_name.append(builder.CreateString('output'+str(i)))

        if isSoftmax == True:
            softmax_name = builder.CreateString('softmax_output')

        # firmware, parameter buffer vector
        firm_vector = builder.CreateByteVector(bytearray(firmware_buf))
        param_vector = builder.CreateByteVector(bytearray(parameter_buf))


        # OperaterCode
        tflite.OperatorCode.OperatorCodeStart(builder)
        tflite.OperatorCode.OperatorCodeAddBuiltinCode(builder, tflite.BuiltinOperator.BuiltinOperator().LNE_EXEC)
        op0 = tflite.OperatorCode.OperatorCodeEnd(builder)

        tflite.OperatorCode.OperatorCodeStart(builder)
        tflite.OperatorCode.OperatorCodeAddBuiltinCode(builder, tflite.BuiltinOperator.BuiltinOperator().SOFTMAX)
        op1 = tflite.OperatorCode.OperatorCodeEnd(builder)
        if isSoftmax == False:
            tflite.Model.ModelStartOperatorCodesVector(builder, 1)
            builder.PrependUOffsetTRelative(op0)
            lne_op = builder.EndVector(1)
        else:
            tflite.Model.ModelStartOperatorCodesVector(builder, 2)
            builder.PrependUOffsetTRelative(op1)
            builder.PrependUOffsetTRelative(op0)
            lne_op = builder.EndVector(2)


        # sub tensor(firmware, parameter, input, output, softmax)
        tflite.Tensor.TensorStartShapeVector(builder, 1)
        builder.PrependInt32(fsize)
        f_shapes = builder.EndVector(1)

        tflite.Tensor.TensorStart(builder)
        tflite.Tensor.TensorAddType(builder, tflite.TensorType.TensorType().UINT8)
        tflite.Tensor.TensorAddBuffer(builder, 1)
        tflite.Tensor.TensorAddShape(builder, f_shapes)
        tflite.Tensor.TensorAddName(builder, firm_name)
        f_tensor = tflite.Tensor.TensorEnd(builder)

        tflite.Tensor.TensorStartShapeVector(builder, 1)
        builder.PrependInt32(psize)
        p_shapes = builder.EndVector(1)

        tflite.Tensor.TensorStart(builder)
        tflite.Tensor.TensorAddType(builder, tflite.TensorType.TensorType().UINT8)
        tflite.Tensor.TensorAddBuffer(builder, 2)
        tflite.Tensor.TensorAddShape(builder, p_shapes)
        tflite.Tensor.TensorAddName(builder, param_name)
        p_tensor = tflite.Tensor.TensorEnd(builder)


	# multi input tensor
        input_shapes=[]
        i_tensor=[]
        for i in range(input_num):
            tflite.Tensor.TensorStartShapeVector(builder, 4)
            builder.PrependInt32(input_shape[i][3])
            builder.PrependInt32(input_shape[i][2])
            builder.PrependInt32(input_shape[i][1])
            builder.PrependInt32(input_shape[i][0])
            input_shapes.append(builder.EndVector(4))

            tflite.Tensor.TensorStart(builder)
            tflite.Tensor.TensorAddBuffer(builder, 3 + i)
            tflite.Tensor.TensorAddName(builder, input_name[i])
            if input_type == "UINT8":
                tflite.Tensor.TensorAddType(builder, tflite.TensorType.TensorType().UINT8)
            elif input_type == "INT8":
                tflite.Tensor.TensorAddType(builder, tflite.TensorType.TensorType().INT8)
            elif input_type == "UINT16":
                tflite.Tensor.TensorAddType(builder, tflite.TensorType.TensorType().UINT16)
            elif input_type == "INT16":
                tflite.Tensor.TensorAddType(builder, tflite.TensorType.TensorType().INT16)
            elif input_type == "FLOAT32":
                tflite.Tensor.TensorAddType(builder, tflite.TensorType.TensorType().FLOAT32)
            else:
                print("not support input type: ", input_type)
                sys.exit(1);
            tflite.Tensor.TensorAddShape(builder, input_shapes[i])
            i_tensor.append(tflite.Tensor.TensorEnd(builder))

        output_shapes=[]
        o_tensor=[]
        tensorout_num = 0
        # multi output tensor
        if isSoftmax == False:
            tensorout_num = output_num
            for i in range(tensorout_num):
                tflite.Tensor.TensorStartShapeVector(builder, 4)
                builder.PrependInt32(output_shape[i][3])
                builder.PrependInt32(output_shape[i][2])
                builder.PrependInt32(output_shape[i][1])
                builder.PrependInt32(output_shape[i][0])
                output_shapes.append(builder.EndVector(4))

                tflite.Tensor.TensorStart(builder)
                tflite.Tensor.TensorAddBuffer(builder, 3 + input_num + i)
                tflite.Tensor.TensorAddName(builder, output_name[i])
                if output_type == "INT8":
                    tflite.Tensor.TensorAddType(builder, tflite.TensorType.TensorType().INT8)
                elif output_type == "INT16":
                    tflite.Tensor.TensorAddType(builder, tflite.TensorType.TensorType().INT16)
                elif output_type == "FLOAT32":
                    tflite.Tensor.TensorAddType(builder, tflite.TensorType.TensorType().FLOAT32)
                tflite.Tensor.TensorAddShape(builder, output_shapes[i])
                o_tensor.append(tflite.Tensor.TensorEnd(builder))
        else:
            tensorout_num = output_num + 1
            for i in range(tensorout_num):
                tflite.Tensor.TensorStartShapeVector(builder, 2)
                builder.PrependInt32(output_shape[0][3])
                builder.PrependInt32(output_shape[0][0])
                output_shapes.append(builder.EndVector(2))

                tflite.Tensor.TensorStart(builder)
                tflite.Tensor.TensorAddBuffer(builder, 3 + input_num + i)
                if i == 0:
                    tflite.Tensor.TensorAddName(builder, output_name[i])
                else:
                    tflite.Tensor.TensorAddName(builder, softmax_name)
                tflite.Tensor.TensorAddType(builder, tflite.TensorType.TensorType().FLOAT32)
                tflite.Tensor.TensorAddShape(builder, output_shapes[i])
                o_tensor.append(tflite.Tensor.TensorEnd(builder))

        # ADD tensors to Subgraph
        tflite.SubGraph.SubGraphStartTensorsVector(builder, 2 + input_num + tensorout_num)
        for i in reversed(range(tensorout_num)):
            builder.PrependUOffsetTRelative(o_tensor[i])
        for i in reversed(range(input_num)):
            builder.PrependUOffsetTRelative(i_tensor[i])
        builder.PrependUOffsetTRelative(p_tensor)
        builder.PrependUOffsetTRelative(f_tensor)
        lne_tensor = builder.EndVector(2 + input_num + tensorout_num)

        '''
        # Dynamic Descriptor for LTE
        # layer output offset
        # output q value
        # param_offset
        tflite.LneExecOptions.LneExecOptionsStartParamOffsetVector(builder, poffset_num)
        for i in range(poffset_num)
            builder.PrependInt32(p_offset[i])
        param_offset = builder.EndVector(poffset_num)

        # param_qvalue
        tflite.LneExecOptions.LneExecOptionsStartParamQVector(builder, pqvalue_num)
        for i in range(pqvalue_num)
            builder.PrependInt32(p_qvalue[i])
        param_qvalue = builder.EndVector(pqvalue_num)
        '''
	# LNE ext op option
        tflite.LneExecOptions.LneExecOptionsStart(builder)
        tflite.LneExecOptions.LneExecOptionsAddFirmwareSize(builder, fsize)
        tflite.LneExecOptions.LneExecOptionsAddParameterSize(builder, psize)
        tflite.LneExecOptions.LneExecOptionsAddIbufferSize(builder, bsize)
        tflite.LneExecOptions.LneExecOptionsAddInputQ(builder, inputq_multi)
        tflite.LneExecOptions.LneExecOptionsAddOutputQ(builder, outputq_multi)
        tflite.LneExecOptions.LneExecOptionsAddScale(builder, scale)
        tflite.LneExecOptions.LneExecOptionsAddMean(builder, tflite.mean_vec.Createmean_vec(builder,
            mean_b, mean_g, mean_r))
        tflite.LneExecOptions.LneExecOptionsAddLneVersion(builder, lne_hw_version)
        tflite.LneExecOptions.LneExecOptionsAddFxp(builder, s_qmode)
        tflite.LneExecOptions.LneExecOptionsAddInFormat(builder, s_idata_format)
        tflite.LneExecOptions.LneExecOptionsAddOutFormat(builder, s_odata_format)
        # Dynamic Descriptor for LTE
        #tflite.LneExecOptions.LneExecOptionsAddParamOffset(builder, param_offset)
        #tflite.LneExecOptions.LneExecOptionsAddParamQ(builder, param_qvalue)
        op0_option = tflite.LneExecOptions.LneExecOptionsEnd(builder)

        # LNE operator inputs / outputs (tensor number)
        tflite.Operator.OperatorStartInputsVector(builder, 2 + input_num)
        for i in reversed(range(input_num -1)):
            builder.PrependInt32(input_num + 1 + i)
        builder.PrependInt32(1)
        builder.PrependInt32(0)
        builder.PrependInt32(2)
        op0_input = builder.EndVector(2 + input_num)

        tflite.Operator.OperatorStartOutputsVector(builder, output_num)
        for i in reversed(range(output_num)):
            builder.PrependInt32(2 + input_num + i)
        op0_output = builder.EndVector(output_num)

        #LNE operator
        tflite.Operator.OperatorStart(builder)
        tflite.Operator.OperatorAddOpcodeIndex(builder, 0) # LNE opcode
        tflite.Operator.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.BuiltinOptions().LneExecOptions)
        tflite.Operator.OperatorAddInputs(builder, op0_input)
        tflite.Operator.OperatorAddOutputs(builder, op0_output)
        tflite.Operator.OperatorAddBuiltinOptions(builder, op0_option)
        lne_op0 = tflite.Operator.OperatorEnd(builder)

        #softmax op option
        tflite.SoftmaxOptions.SoftmaxOptionsStart(builder)
        tflite.SoftmaxOptions.SoftmaxOptionsAddBeta(builder, 1)
        op1_option = tflite.SoftmaxOptions.SoftmaxOptionsEnd(builder)

        #softmax inputs / outputs (tensor number)
        tflite.Operator.OperatorStartInputsVector(builder, 1)
        builder.PrependInt32(2 + input_num)
        op1_input = builder.EndVector(1)

        tflite.Operator.OperatorStartOutputsVector(builder, 1)
        builder.PrependInt32(2 + input_num + 1)
        op1_output = builder.EndVector(1)

        #softmax operator
        tflite.Operator.OperatorStart(builder)
        tflite.Operator.OperatorAddOpcodeIndex(builder, 1) # softmax opcode
        tflite.Operator.OperatorAddBuiltinOptionsType(builder, tflite.BuiltinOptions.BuiltinOptions().SoftmaxOptions)
        tflite.Operator.OperatorAddInputs(builder, op1_input)
        tflite.Operator.OperatorAddOutputs(builder, op1_output)
        tflite.Operator.OperatorAddBuiltinOptions(builder, op1_option)
        lne_op1 = tflite.Operator.OperatorEnd(builder)

        if isSoftmax == False:
            tflite.SubGraph.SubGraphStartOperatorsVector(builder, 1)
            builder.PrependUOffsetTRelative(lne_op0)
            lne_operator = builder.EndVector(1)
        else:
            tflite.SubGraph.SubGraphStartOperatorsVector(builder, 2)
            builder.PrependUOffsetTRelative(lne_op1)
            builder.PrependUOffsetTRelative(lne_op0)
            lne_operator = builder.EndVector(2)


        # sub inputs: tensor number
        tflite.SubGraph.SubGraphStartInputsVector(builder, input_num)
        for i in reversed(range(input_num)):
            builder.PrependInt32(2 + i)
        lne_input = builder.EndVector(input_num)

        # sub ouputs: tensor number
        if isSoftmax == False:
            tflite.SubGraph.SubGraphStartOutputsVector(builder, output_num)
            for i in reversed(range(output_num)):
                builder.PrependInt32(2 + input_num + i)
            lne_output = builder.EndVector(output_num)
        else:
            tflite.SubGraph.SubGraphStartOutputsVector(builder, 1)
            builder.PrependInt32(2 + input_num + output_num)
            lne_output = builder.EndVector(1)

        # Subgraph
        tflite.SubGraph.SubGraphStart(builder)
        tflite.SubGraph.SubGraphAddTensors(builder, lne_tensor)
        tflite.SubGraph.SubGraphAddOperators(builder, lne_operator)
        tflite.SubGraph.SubGraphAddInputs(builder, lne_input)
        tflite.SubGraph.SubGraphAddOutputs(builder, lne_output)
        tflite.SubGraph.SubGraphAddName(builder, sub_desc)
        subgraph0 = tflite.SubGraph.SubGraphEnd(builder)

        tflite.Model.ModelStartSubgraphsVector(builder, 1)
        builder.PrependUOffsetTRelative(subgraph0)
        lne_subgraph = builder.EndVector(1)

        # BUFFER Define
        # free buffer (defualt = 1)
        tflite.Buffer.BufferStart(builder)
        buf_free = tflite.Buffer.BufferEnd(builder)

        # LNE input / output buf
        buf_input = []
        for i in range(input_num):
            tflite.Buffer.BufferStart(builder)
            buf_input.append(tflite.Buffer.BufferEnd(builder))

        buf_output = []
        for i in range(output_num):
            tflite.Buffer.BufferStart(builder)
            buf_output.append(tflite.Buffer.BufferEnd(builder))

        # for softmax buf
        tflite.Buffer.BufferStart(builder)
        buf_softmax = tflite.Buffer.BufferEnd(builder)

        # firmware / parameter
        tflite.Buffer.BufferStart(builder)
        tflite.Buffer.BufferAddData(builder,firm_vector)
        buf_firmware = tflite.Buffer.BufferEnd(builder)

        tflite.Buffer.BufferStart(builder)
        tflite.Buffer.BufferAddData(builder, param_vector)
        buf_parameter = tflite.Buffer.BufferEnd(builder)

        buf_num = 0
        if isSoftmax == False:
            buf_num = 3 + input_num + output_num
        else:
            buf_num = 3 + input_num + output_num + 1

        tflite.Model.ModelStartBuffersVector(builder, buf_num)
        if isSoftmax == True:
            builder.PrependUOffsetTRelative(buf_softmax)
        for i in range(output_num):
            builder.PrependUOffsetTRelative(buf_output[i])
        for i in range(input_num):
            builder.PrependUOffsetTRelative(buf_input[i])
        builder.PrependUOffsetTRelative(buf_parameter)
        builder.PrependUOffsetTRelative(buf_firmware)
        builder.PrependUOffsetTRelative(buf_free)
        lne_buffers = builder.EndVector(buf_num)

        # MODEL root
        tflite.Model.ModelStart(builder)
        tflite.Model.ModelAddVersion(builder, lne_version)
        tflite.Model.ModelAddOperatorCodes(builder, lne_op)
        tflite.Model.ModelAddSubgraphs(builder, lne_subgraph)
        tflite.Model.ModelAddBuffers(builder, lne_buffers)
        tflite.Model.ModelAddDescription(builder, lne_desc)

        lne_orc = tflite.Model.ModelEnd(builder)

        builder.Finish(lne_orc)
        buf = builder.Output()

        temp = os.path.basename(runscript)
        out_filename = temp[:-4]
        print("-------------------------------------------------")
        print("LNE model name : {}.lne".format(out_filename))
        print("-------------------------------------------------")

        with open(out_filename + ".lne", "wb") as o:
            o.write(buf)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', help="firmware path(ex> rcrfirm.bin)")
    parser.add_argument('-p', help="parameter path(ex> nmp_net_xxx_parameter.bin)")
    parser.add_argument('-r', help="runscript path(ex> run_nmp_net_xxx)")
    parser.add_argument('-i', default = "None", help="input/output info.txt(ex> inout_info.txt)")
    parser.add_argument('-m', nargs="*", help="mean values(channel order) ex> 111.5 117.3 115.6, default: 0.0 0.0 0.0", default= [0.0, 0.0, 0.0], type=float)
    parser.add_argument('-s', help="scale ex>0.017, defaut: 1.0", default=1.0, type=float)
    parser.add_argument('-it', help="input type ex> INT8/UINT8/INT16/FLOAT32, default: FLOAT32", default="FLOAT32")
    parser.add_argument('-ot', help="output type ex> INT8/INT16/FLOAT32, default: FLOAT32", default="FLOAT32")
    parser.add_argument('-idf', help="data format ex> NHWC/NCHW, default: NHWC", default="NHWC")
    parser.add_argument('-odf', help="data format ex> NHWC/NCHW, default: NHWC", default="NHWC")
    parser.add_argument('-v', help="lne version ex> 240, 250", default="250", type=int)
    parser.add_argument('-w', help="with post processing ex> softmax", default="None")
    args = parser.parse_args()
    main(args.f, args.p, args.r, args.i, args.m, args.s, args.it, args.ot, args.idf, args.odf, args.w, args.v)
