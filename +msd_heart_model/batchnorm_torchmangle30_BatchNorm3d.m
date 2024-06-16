classdef batchnorm_torchmangle30_BatchNorm3d < nnet.layer.Layer & nnet.layer.Formattable & ...
        nnet.layer.AutogeneratedFromPyTorch & nnet.layer.Acceleratable
    %batchnorm_torchmangle30_BatchNorm3d Auto-generated custom layer
    % Auto-generated by MATLAB on 16-Jun-2024 19:15:24
    
    properties (Learnable)
        % Networks (type dlnetwork)
        
    end
    
    properties
        % Non-Trainable Parameters
        Constant_21
        Constant_22
        Constant_23
        
        
        Param_running_var
        Param_running_mean
    end
    
    properties (Learnable)
        % Trainable Parameters
        Param_bias
        Param_weight
    end
    
    methods
        function obj = batchnorm_torchmangle30_BatchNorm3d(Name, Type, InputNames, OutputNames)
            obj.Name = Name;
            obj.Type = Type;
            obj.NumInputs = 1;
            obj.NumOutputs = 1;
            obj.InputNames = InputNames;
            obj.OutputNames = OutputNames;
        end
        
        function [batchnorm_input_1] = predict(obj,batchnorm_argument1_1)
            
            %Use the input format inferred by the importer to permute the input into reverse-PyTorch dimension order
            [batchnorm_argument1_1, batchnorm_argument1_1_format] = msd_heart_model.ops.permuteToReversePyTorch(batchnorm_argument1_1, 'BCSSS', 5);
            [batchnorm_argument1_1] = struct('value', batchnorm_argument1_1, 'rank', int64(5));
            
            [batchnorm_input_1] = tracedPyTorchFunction(obj,batchnorm_argument1_1,false,"predict");
            
            
            [batchnorm_input_1] = msd_heart_model.ops.labelWithPropagatedFormats(batchnorm_input_1, "BCSSS");
            batchnorm_input_1 = batchnorm_input_1.value ;
            
        end
        
        
        
        function [batchnorm_input_1] = forward(obj,batchnorm_argument1_1)
            
            %Use the input format inferred by the importer to permute the input into reverse-PyTorch dimension order
            [batchnorm_argument1_1, batchnorm_argument1_1_format] = msd_heart_model.ops.permuteToReversePyTorch(batchnorm_argument1_1, 'BCSSS', 5);
            [batchnorm_argument1_1] = struct('value', batchnorm_argument1_1, 'rank', int64(5));
            
            [batchnorm_input_1] = tracedPyTorchFunction(obj,batchnorm_argument1_1,true,"forward");
            
            
            [batchnorm_input_1] = msd_heart_model.ops.labelWithPropagatedFormats(batchnorm_input_1, "BCSSS");
            batchnorm_input_1 = batchnorm_input_1.value ;
            
        end
        
        
        
        function [batchnorm_input_1] = tracedPyTorchFunction(obj,batchnorm_argument1_1,isForward,predict)
            
            [Constant_21] = msd_heart_model.ops.makeStructForConstant(int64(obj.Constant_21), int64(0), "Typed");
            [Constant_22] = msd_heart_model.ops.makeStructForConstant(single(obj.Constant_22), int64(0), "Typed");
            [Constant_23] = msd_heart_model.ops.makeStructForConstant(single(obj.Constant_23), int64(0), "Typed");
            GetAttr_runningvar_1 = obj.Param_running_var;
            
            [GetAttr_runningvar_1] = struct('value', dlarray(GetAttr_runningvar_1,'UU'), 'rank', 1);
            
            GetAttr_runningmean_1 = obj.Param_running_mean;
            
            [GetAttr_runningmean_1] = struct('value', dlarray(GetAttr_runningmean_1,'UU'), 'rank', 1);
            
            GetAttr_bias_1 = obj.Param_bias;
            
            [GetAttr_bias_1] = struct('value', dlarray(GetAttr_bias_1,'UU'), 'rank', 1);
            
            GetAttr_weight_1 = obj.Param_weight;
            
            [GetAttr_weight_1] = struct('value', dlarray(GetAttr_weight_1,'UU'), 'rank', 1);
            
            [batchnorm_input_1] = msd_heart_model.ops.pyBatchNorm(batchnorm_argument1_1, GetAttr_weight_1, GetAttr_bias_1, GetAttr_runningmean_1, GetAttr_runningvar_1, Constant_22, Constant_23, isForward);
        end
        
    end
end

