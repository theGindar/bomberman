import matplotlib
from matplotlib import pyplot as plt
from datetime import datetime

def plot_rewards(self):
    
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    print(f'total reward history length: {len(self.total_reward_history)}')
    #plt.plot(range(self.current_episode_num-1), self.total_reward_history)
    plt.plot([1]], [1])
    plt.ylim(ymin=0)
    current_datetime = datetime.now()
    plt.savefig(f'./plots/rewards_{str(current_datetime)}.png')
    #plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    #plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    #plt.pause(1)
    #display.clear_output(wait=True)
    #display.display(plt.gcf())