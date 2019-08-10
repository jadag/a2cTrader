import numpy as np
import matplotlib.pyplot as plt
import random


class StockMarketEnv:

    portfolio_len = 12
    capital_record = []
    action_space = 1

    trans_cost = 0.000
    devalue = 0.9
    highest_price = 0
    time = 0
    
    def __init__(self, set_quotes_, start_time, time_range, channels, sub_title, testing_):
        self.portfolio_len = len(set_quotes_)
        # collect_quotes = []
        # for quotes in set_quotes_:
        #     collect_quotes.append(quotes)
        
        self.stock_history = np.array(set_quotes_)

        self.period_length = self.stock_history.shape[1] 
        self.price_changes = []
        self.channels = channels
        self.max_time = self.stock_history.shape[1]
        self.begin_time = start_time
        self.state_sz = time_range

        fiat_0 = 1

        self.fiat = np.array([[0.0] * channels] * (1 + time_range))
        for i in range(time_range, 0, -1):
            self.fiat[i] = fiat_0
            fiat_0 = fiat_0 * self.devalue

        self.action_space = self.portfolio_len + 1
        self.portfolio = np.array([0] * self.action_space)
        # self.portfolio[-1] = 1 #last element is fiat currency
        self.capital = 1
        self.portfolio = np.array([0.01] * self.action_space)
        self.portfolio[-1] = 0.98  # last element is fiat currency

        self.pre_process()
        self.fig = plt.figure()
        self.sub_title = sub_title
    
    def show_results(self, actions, period_length):
        # fig = plt.figure()
        self.fig.clf()
        self.fig.suptitle(self.sub_title, fontsize=16)
        # plt.clf()
    
        ax = self.fig.add_subplot(2, 1, 1)
        ax2 = self.fig.add_subplot(2, 1, 2)
        ax2.cla()
        colors = ['r', 'g', 'y', 'b', 'k']
        # env.plot_results(ax2, colors)
        ax.cla()
        actions = np.array(actions)
        capital = self.get_quotes(period_length)
        capital = np.array(capital)
        # ax.plot(actions)
        # ax.plot(capital)
    
        for i in range(actions.shape[1]):
            ax.plot(actions[:, i], label=colors[i])
        # ax.legend()
        print('actionss', actions.shape, 'cap ', capital.shape)
        plt.show(block=False)
        for i in range(capital.shape[0]):
#             print(capital[i,self.start_time:self.time])
            ax2.plot(capital[i], label=colors[i])
    
        ax2.plot(self.capital_record, label='k')
            # ax.legend()
        plt.show(block=False)
        plt.pause(0.1)
         
    def render(self, stock):
        graph = self.stock_history[stock, self.time: self.time + self.state_sz]
        plt.clf()
        plt.plot(graph)
        plt.draw()
        plt.pause(0.1)

    def pre_process(self):
        print('preprocess')
        for quote in self.stock_history:
            print('stock_history ', self.stock_history.shape)
            if self.channels < 2:
                price_change = (quote[1:] - quote[:-1]) / quote[:-1]
            else:
                price_change = (quote[1:, :] - quote[:-1, :]) / quote[:-1, :]
            self.price_changes.append(price_change)
            print('price_change ', np.array(self.price_changes).shape)
        self.price_changes = np.array(self.price_changes)

    def normalize_quote(self):
        quote = np.copy(self.stock_history[:, self.time: self.time + self.state_sz])

        for i in range(len(quote)):
            quote[i] = (quote[i] - quote[i, 0]) / quote[i, 0]

        return dict({'quote' :quote , 'position':self.portfolio})

    def profit(self, old_capital):
        if self.channels < 2:
            price_change = self.price_changes[:, self.time]
        else:
            price_change = self.price_changes[:, self.time, 2]

        gains = np.sum(self.portfolio[:-1] * price_change)  # -self.portfolio[-1]*(1-self.devalue)

        self.capital += gains * self.capital
        # if self.capital < 0.01 or self.capital != self.capital:
        #     self.capital = 1
        reward = (self.capital - old_capital) / old_capital
#         reward = self.capital -1
        return reward * 3

    def reset(self, start_time=None):
        if self.time + 2000 >= self.max_time:
            self.period_length = self.max_time
            self.start_time = 0
        elif start_time is not None:
            self.start_time = start_time
        else:
            self.start_time = self.time  # random.randint( self.state_sz+1, self.max_time -  self.period_length - 200)

        self.time = self.start_time
#         print('self.start_time ',self.start_time)
        self.capital = 1  # everybody gets a second chance
        self.capital_record = []
        norm_quote = self.normalize_quote()

        return norm_quote

    def make_step(self, action):
        old_capital = self.capital
        
        transaction_cost = self.capital * self.trans_cost * np.sum((np.abs(self.portfolio - action)))
        self.capital -= transaction_cost
        self.portfolio = action

        new_price = self.stock_history[:, self.time]
        old_price = self.stock_history[:, self.time - 1]
        change =(old_price - new_price)/old_price

        done = False
        self.time += 1

        if (self.time + self.state_sz + 1) >= self.max_time:
            self.time = self.begin_time
            done = True
        
        capital_change = change[:,0].dot(self.portfolio[:-1])
        self.capital += self.capital*capital_change
        period_quote = self.stock_history[:, self.start_time:self.time]

#         reward = self.profit(old_capital)

        self.capital_record.append(self.capital)
#         reward = 100*(self.capital -1)/(self.time-self.start_time)
        # reward = (np.random.randint(20, size=1)-10)[0]
        norm_quote = self.normalize_quote()
        max_value = np.max(norm_quote['quote'])
        reward = capital_change*100       
        # reward = self.capital -1
        return norm_quote , reward , done

    def get_quotes(self, period_length):
        close_quote = self.stock_history[:, self.start_time: self.start_time + period_length, 2]
        # quote = quote
        for i in range(close_quote.shape[0]):
            close_quote[i] = close_quote[i] / close_quote[i, 0]

#         print('get quote ', close_quote)
        return close_quote

    def plot_results(self, ax, colors):

        performances = []
        i = 0
        for quote in self.stock_history[:, self.start_time : self.start_time + self.period_length]:
            new_quote = quote / quote[0, 2]
            performances.append(new_quote[-1])
            ax.plot(new_quote, label=colors[i])
            i += 1
        ax.plot(self.capital_record, '--r')

        print('capital left', self.capital, 'performances ', performances)
