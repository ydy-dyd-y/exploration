function [data,latents] = thompson_sim(param,data)   % 각 subject의 data인 data(s) 가 들어감
    
    % One-dimensional Kalman filter.
    
    % parameters
    q = param(1);           % reward variance : 10 / 10
    q1 = param(2);          % 10 / 100 
    q2 = param(3);          % 0(fixed value in exp1) / 100
    
    for n = 1:length(data.block)
        
        % initialization at the start of each block
        if n == 1 || data.block(n)~=data.block(n-1)
            m = [0 0];  % posterior mean
            s = [q1 q2];  % posterior variance
        end
        
        % choice
        p = normcdf(m(1)-m(2)/sqrt(s(1)+s(2)));
        
        try
            c = data.c(n);
            r = data.r(n);
        catch
            if rand < p
                c = 1;  % 즉 p의 확률로 Arm 1을 선택
            else
                c = 2;
            end
            
            % feedback
            r = data.R(n,c);
            
            data.c(n,1) = c;
            data.r(n,1) = r;
        end
        
        % store latents
        if nargout > 1
            latents.m(n,:) = m;
            latents.s(n,:) = s;
            latents.p(n,1) = p;
        end
        
        % update
        k = s(c)/(s(c)+q);         % Kalman gain
        err = r - m(c);            % prediction error
        m(c) = m(c) + k*err;       % posterior mean
        s(c) = s(c) - k*s(c);      % posterior variance
        
    end
