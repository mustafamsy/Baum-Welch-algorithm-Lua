-- Implementation of the Baum-Welch algorithm in Lua

-- Define the Baum-Welch algorithm
function baum_welch(obs, states, start_p, trans_p, emit_p)
    local n_states = #states
    local n_obs = #obs

    -- Initialize alpha and beta matrices
    local alpha = {}
    local beta = {}
    for i = 1, n_states do
        alpha[i] = {}
        beta[i] = {}
        alpha[i][1] = start_p[i] * emit_p[i][obs[1]]
        beta[i][n_obs] = 1
    end

    -- Compute alpha matrix
    for t = 2, n_obs do
        for i = 1, n_states do
            alpha[i][t] = 0
            for j = 1, n_states do
                alpha[i][t] = alpha[i][t] + (alpha[j][t-1] * trans_p[j][i])
            end
            alpha[i][t] = alpha[i][t] * emit_p[i][obs[t]]
        end
    end

    -- Compute beta matrix
    for t = n_obs-1, 1, -1 do
        for i = 1, n_states do
            beta[i][t] = 0
            for j = 1, n_states do
                beta[i][t] = beta[i][t] + (trans_p[i][j] * emit_p[j][obs[t+1]] * beta[j][t+1])
            end
        end
    end

    -- Compute gamma and di-gamma matrices
    local gamma = {}
    local di_gamma = {}
    for t = 1, n_obs-1 do
        local denom = 0
        gamma[t] = {}
        di_gamma[t] = {}
        for i = 1, n_states do
            gamma[t][i] = alpha[i][t] * beta[i][t]
            denom = denom + gamma[t][i]
        end
        for i = 1, n_states do
            gamma[t][i] = gamma[t][i] / denom
            di_gamma[t][i] = {}
            for j = 1, n_states do
                di_gamma[t][i][j] = (alpha[i][t] * trans_p[i][j] * emit_p[j][obs[t+1]] * beta[j][t+1]) / denom
            end
        end
    end

    -- Update start probabilities
    for i = 1, n_states do
        start_p[i] = gamma[1][i]
    end

    -- Update transition probabilities
    for i = 1, n_states do
        local denom = 0
        for t = 1, n_obs-1 do
            denom = denom + gamma[t][i]
        end
        for j = 1, n_states do
            local numer = 0
            for t = 1, n_obs-1 do
                numer = numer + di_gamma[t][i][j]
            end
            trans_p[i][j] = numer / denom
        end
    end

    -- Update emission probabilities
    for i = 1, n_states do
        local denom = 0
        for t=1,n_obs do 
            denom=denom+gamma[t][i]
        end 
        for j=1,#obs[0]+2 do 
            local numer=0 
            for t=1,n_obs do 
                if obs[t]==j then 
                    numer=numer+gamma[t][i]
                end 
            end 
            emit_p[i][j]=numer/denom 
        end 
    end 

end

-- Example usage of the Baum-Welch algorithm with a hidden Markov model (HMM)
local obs_seq={2,3,2}
local states={"A","B"}
local start_prob={0.5,0.5}
local trans_prob={{0.7,0.3},{0.4,0.6}}
local emit_prob={{0.2,0.4,0.4
