classdef aten___convolution7 < nnet.layer.Layer & nnet.layer.Formattable & ...
        nnet.layer.AutogeneratedFromPyTorch & nnet.layer.Acceleratable
    %aten___convolution7 Auto-generated custom layer
    % Auto-generated by MATLAB on 16-Jun-2024 19:15:24
    
    properties (Learnable)
        % Networks (type dlnetwork)
        
    end
    
    properties
        % Non-Trainable Parameters
        convolution_6
        convolution_29
        convolution_30
        convolution_31
        convolution_12
        convolution_32
        convolution_9
        convolution_14
        convolution_141
        convolution_121
        convolution_122
        
        
        
    end
    
    properties (Learnable)
        % Trainable Parameters
        Param_weight
    end
    
    methods
        function obj = aten___convolution7(Name, Type, InputNames, OutputNames)
            obj.Name = Name;
            obj.Type = Type;
            obj.NumInputs = 1;
            obj.NumOutputs = 1;
            obj.InputNames = InputNames;
            obj.OutputNames = OutputNames;
        end
        
        function [convolution_input_1] = predict(obj,convolution_argument1_1)
            
            %Use the input format inferred by the importer to permute the input into reverse-PyTorch dimension order
            [convolution_argument1_1, convolution_argument1_1_format] = msd_heart_model.ops.permuteToReversePyTorch(convolution_argument1_1, 'BCSSS', 5);
            [convolution_argument1_1] = struct('value', convolution_argument1_1, 'rank', int64(5));
            
            [convolution_input_1] = tracedPyTorchFunction(obj,convolution_argument1_1,false,"predict");
            
            
            [convolution_input_1] = msd_heart_model.ops.labelWithPropagatedFormats(convolution_input_1, "BCSSS");
            convolution_input_1 = convolution_input_1.value ;
            
        end
        
        
        
        function [convolution_input_1] = forward(obj,convolution_argument1_1)
            
            %Use the input format inferred by the importer to permute the input into reverse-PyTorch dimension order
            [convolution_argument1_1, convolution_argument1_1_format] = msd_heart_model.ops.permuteToReversePyTorch(convolution_argument1_1, 'BCSSS', 5);
            [convolution_argument1_1] = struct('value', convolution_argument1_1, 'rank', int64(5));
            
            [convolution_input_1] = tracedPyTorchFunction(obj,convolution_argument1_1,true,"forward");
            
            
            [convolution_input_1] = msd_heart_model.ops.labelWithPropagatedFormats(convolution_input_1, "BCSSS");
            convolution_input_1 = convolution_input_1.value ;
            
        end
        
        
        
        function [convolution_input_1] = tracedPyTorchFunction(obj,convolution_argument1_1,isForward,predict)
            
            convolution_weight_1 = obj.Param_weight;
            
            [convolution_weight_1] = struct('value', dlarray(convolution_weight_1,'UUUUU'), 'rank', 5);
            
            [convolution_6] = msd_heart_model.ops.makeStructForConstant(double(obj.convolution_6), int64(0), "Typed");
            [convolution_29] = msd_heart_model.ops.makeStructForConstant(int64(obj.convolution_29), int64(1), "Typed");
            [convolution_30] = msd_heart_model.ops.makeStructForConstant(int64(obj.convolution_30), int64(1), "Typed");
            [convolution_31] = msd_heart_model.ops.makeStructForConstant(int64(obj.convolution_31), int64(1), "Typed");
            [convolution_12] = msd_heart_model.ops.makeStructForConstant(int64(obj.convolution_12), int64(0), "Typed");
            [convolution_32] = msd_heart_model.ops.makeStructForConstant(int64(obj.convolution_32), int64(1), "Typed");
            [convolution_9] = msd_heart_model.ops.makeStructForConstant(int64(obj.convolution_9), int64(0), "Typed");
            [convolution_14] = msd_heart_model.ops.makeStructForConstant(int64(obj.convolution_14), int64(0), "Typed");
            [convolution_141] = msd_heart_model.ops.makeStructForConstant(int64(obj.convolution_141), int64(0), "Typed");
            [convolution_121] = msd_heart_model.ops.makeStructForConstant(int64(obj.convolution_121), int64(0), "Typed");
            [convolution_122] = msd_heart_model.ops.makeStructForConstant(int64(obj.convolution_122), int64(0), "Typed");
            [convolution_input_1] = msd_heart_model.ops.pyConvolution(convolution_argument1_1, convolution_weight_1, convolution_6, convolution_29, convolution_30, convolution_31, convolution_12, convolution_32, convolution_9);
        end
        
    end
end

