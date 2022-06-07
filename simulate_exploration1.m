function simulate_exploration1
    
    nSubjects = 500;
    nBlocks = 10;       % number of blocks
    N = 20;             % block length
    q0 = 10;           % prior variance
    q = 10;             % observation variance
    b = 1;
    tau = 1;
    
    % generate data
    for s = 1:nSubjects
        data(s).R = []; data(s).block = [];
        for block = 1:nBlocks
            data(s).mu = [data(s).mu; repmat(normrnd(0,sqrt(q0)),N,1)];
            data(s).R = [data(s).R; normrnd(data(s).mu,sqrt(q))];
            data(s).block = [data(s).block; zeros(N,1)+block];    % 두 array를 ;로 연결했으므로 최종적인 array는 세로로 늘어선 200*1크기의 array가 됨
        end
        
        % simulate algorithms
        [data_ucb(s), results(1).latents(s)] = ucb_sim([q q0 0 b tau],data(s));
        [data_thompson(s), results(2).latents(s)] = thompson_sim([q q0 0],data(s));
        [data_hybrid(s), results(3).latents(s)] = hybrid_sim([q q0 0 b],data(s));
    end
    
    save results_sim1 results data_ucb data_thompson data_hybrid
