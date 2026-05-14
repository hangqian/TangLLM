function validateGradients(fun, Parameters, Gradients, pNames)
% 高精度数值导数检验模块 (5-Point Stencil Gradient Check)
fprintf('\n启动高精度数值导数检验 (测试各矩阵首尾元素) \n');
delta = 1e-5; % 对于 5点模板，1e-4 或 1e-5 是双精度浮点数下的最优步长
for pIdx = 1:length(pNames)
    name = pNames{pIdx};
    param_matrix = Parameters.(name);
    grad_matrix = Gradients.(name);

    % 只抽取首个(1)和末尾(end)元素的线性索引
    test_indices = [1, numel(param_matrix)];

    for i = 1:length(test_indices)
        idx = test_indices(i);
        orig_val = param_matrix(idx);

        % 计算 +2*delta
        P_temp = Parameters; P_temp.(name)(idx) = orig_val + 2*delta;
        lossUp2 = fun(P_temp);
        
        % 计算 +1*delta
        P_temp = Parameters; P_temp.(name)(idx) = orig_val + delta;
        lossUp1 = fun(P_temp);
        
        % 计算 -1*delta
        P_temp = Parameters; P_temp.(name)(idx) = orig_val - delta;
        lossDn1 = fun(P_temp);

        % 计算 -2*delta
        P_temp = Parameters; P_temp.(name)(idx) = orig_val - 2*delta;
        lossDn2 = fun(P_temp);

        % 5点模板求数值导数
        gradFD = (-lossUp2 + 8*lossUp1 - 8*lossDn1 + lossDn2) / (12 * delta);

        % 提取手写的解析导数
        gradAnalytic = grad_matrix(idx);

        % 计算相对误差 (Relative Error)，用以屏蔽参数绝对大小带来的影响
        abs_err = abs(gradAnalytic - gradFD);
        rel_err = abs_err / (abs(gradAnalytic) + abs(gradFD) + 1e-8);

        idx_str = '首个';
        if i == 2, idx_str = '末尾'; end

        % 打印对比报告
        fprintf('参数 %-18s (%s) | 解析: %10.6e | 数值: %10.6e | 相对误差: %.2e\n', ...
            name, idx_str, gradAnalytic, gradFD, rel_err);
    end
end
fprintf('数值导数检验完成！程序继续往下执行...\n\n');
end

