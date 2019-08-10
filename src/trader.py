import numpy as np

from multiprocessing import Process,  Pipe

import matplotlib.pyplot as plt
import torch


from stockmarket import StockMarketEnv
from A2C import A2C, TestBot
from get_data import load_data, download, create_data


class Trader:
    
    def __init__(self,use_test_data , asset_channels, time_range, train_period, test_period):
        
        self.train_period = train_period
        self.test_period = test_period
        if use_test_data:
            quotes_set = create_data(asset_channels)
        else:
            quotes_set = load_data(get_file=True)
            
        test_size = int(0.15 * quotes_set.shape[1])
        test_quotes = np.array(quotes_set[:, -test_size:, :])
        train_quotes = np.array(quotes_set[:, :-test_size, :])
        self.env = StockMarketEnv(train_quotes, 10, time_range, asset_channels,  sub_title = 'training result',testing_=use_test_data)

        n_x = time_range
        n_y = self.env.action_space
        n_pos = self.env.portfolio_len
        model_file_path = None#'output/crypto_trader_test4.ckpt'
        nr_assets = quotes_set.shape[0]
        self.bot = TestBot(input_sz=asset_channels, act_sz=n_y,nr_assets=nr_assets,save_path=model_file_path)
        self.bot = A2C(input_sz=asset_channels, act_sz=n_y,nr_assets=nr_assets,save_path=model_file_path)
#         self.score_fig = plt.figure()
    
    def loop_env(self, start_time, trade_period):
    
        observation = self.env.reset(start_time)
    
        done = False
        sum_reward = 0
        # sum_rewards =[]
        T = 0.
        # test_outputs = []
        eps_capital = []
        rewards =[]
        while T < trade_period and not done:
    
            action = self.bot.get_action(observation)

            next_observation, reward, done = self.env.make_step(action)
            rewards.append(reward)
            if T > trade_period :
                done = True
                
                if self.env.capital > 1:
                    reward = 1
                else:
                    reward = 0
                
            self.bot.store_state(observation, next_observation, action, reward, done)
            sum_reward = sum_reward + reward
            observation = next_observation
            T += 1
      
            if done:
                break
    
            eps_capital.append(action)
    
        print('action ',action, 'done ', done,' remaining capital ', self.env.capital, 'sum_reward ', sum_reward)
        # sum_rewards.append(sum_reward)
        # market = env.get_result()
#         plt.plot(rewards)
#         plt.show(block=False)
#         plt.pause(0.1)
        end_val = observation['quote'][-1]
        # env.reset()
        return eps_capital
    
    
    
    def run_agents(self, parent_bot, nr_agents):
        pipes = []
        processes = []
        
        for i in range(nr_agents):
            parent_conn, child_conn = Pipe()
            p = Process(target=parent_bot.train, args=(child_conn,))
    
    def run(self, epochs):
        train_score = []
        testScore = []
        trade_period = 100
        for i_episode in range(epochs):
            
            actions = self.loop_env(None, trade_period)
            
            self.env.show_results(actions, trade_period)
            self.bot.train_agent()
            
            self.bot.reset()
            if i_episode % 1 == 0:
                
                profit = self.env.capital - 1
                testScore.append(profit)
    
#                 plt.plot(testScore)
#                 plt.show(block= False)
                self.bot.reset()
                
def main():
    # download()

    channels = 3
    trade_period = 600
    test_period = 16000
    time_range = 50 #length historic data

    use_test_data = True

    crypto_ai = Trader(use_test_data , channels, time_range, trade_period, test_period)
    
    crypto_ai.run(epochs=2000)
    

if __name__ == '__main__':
    main()