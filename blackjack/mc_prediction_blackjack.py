#!/Users/huynguyen/miniforge3/envs/math/bin/python3


"""

    Name: HUY NGUYEN
    *BLACKJACK PROBLEM
    *FIGURE 5.1 IN THE BOOK
    *Assume that cards are drawn from an infinite deck (WITH REPLACEMENT)
    *First-visit MC on-policy prediction, for estimating V with given INPUT POLICY.

"""


from tqdm import tqdm
import numpy as np


class PlayerFrame(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.cards = []
        self.sum = 0
        self.num_aces = 0

    def set_sum(self, val):
        self.sum = val

    def get_sum(self):
        return self.sum

    def get_num_aces(self):
        return self.num_aces
    
    def set_num_aces(self, num_aces):
        self.num_aces = num_aces

    def add_card(self, card):
        self.cards.append(card)

    def get_card_list(self):
        return self.cards


class BlackjackDealer(PlayerFrame):
    def policy(self):
        return 1 if self.sum >= 12 and self.sum < 17 else 0

    def deal_card(self):  # Get a random card from uniform dist WITH REPLACEMENT
        return np.random.randint(low=1, high=14)

    def get_upcard(self):
        return self.cards[0]  # First card is upcard

    def init_state(self):
        while len(self.cards) < 2:  # Dealer needs to have at least 2 cards at init state.
            card = self.deal_card()  # Draw a card
            self.cards.append(card)  # Add card to dealer's hand
            (card_is_ace, card_value) = evaluate_card(card)
            self.num_aces += 1 if card_is_ace else 0

            # Set dealer's current sum
            if not card_is_ace or (card_is_ace and self.sum+11 <= 21):
                self.sum += card_value
            else:
                self.sum = self.sum + 1


class BlackjackPlayer(PlayerFrame):
    def policy(self):
        return 1 if self.sum >= 12 and self.sum < 20 else 0

    def init_state(self, dealer):
        while self.sum < 12:  # Player needs to have at least cumulative sum of 12.
            card = dealer.deal_card()  # Draw a card
            self.cards.append(card)  # Add card to player's hand
            (card_is_ace, card_value) = evaluate_card(card)
            self.num_aces += 1 if card_is_ace else 0

            # Set player's current sum
            if not card_is_ace or (card_is_ace and self.sum+11 <= 21):
                self.sum += card_value
            else:
                self.sum = self.sum + 1


def evaluate_card(card):  # Evaluate card and return tuple (card_is_ace, card_value).
    return (False, min(10, card)) if card != 1 else (True, 11)


def simulation(player=None, dealer=None):  # Monte Carlo Sampling, return final_reward, player_history
    assert player is not None or dealer is not None, 'INVALID INPUT, CANNOT RUN SIMULATION.'

    player_history = []  # List of tuple (player's sum, dealer's upcard, number of aces from player)

    while True:  # Player's turn
        # Add to history for sampling average (MC Sampling/MC Method)
        player_history.append((player.get_sum(), dealer.get_upcard(), player.get_num_aces()))

        action = player.policy()  # Player's decision, 1 is HIT (continue), 0 is STICK (stop).
        if action == 0:
            break

        card = dealer.deal_card()  # Draw a card
        player.add_card(card)  # Add card to player's hand
        (card_is_ace, card_value) = evaluate_card(card)
        if card_is_ace:
            player.set_num_aces(player.get_num_aces()+1)
        
        # Set player's current sum
        if not card_is_ace or (card_is_ace and player.get_sum()+11 <= 21):
            player.set_sum(player.get_sum() + card_value)
        else:
            player.set_sum(player.get_sum()+1)
        
        if player.get_sum() > 21:  # Player is BUSTED!
            return -1, player_history
        
    while True:  # Dealer's turn
        action = dealer.policy()  # Dealer's decision, 1 is HIT (continue), 0 is STICK (stop).
        if action == 0:
            break

        card = dealer.deal_card()  # Draw a card
        dealer.add_card(card)  # Add card to dealer's hand
        (card_is_ace, card_value) = evaluate_card(card)
        if card_is_ace:
            dealer.set_num_aces(dealer.get_num_aces()+1)
        
        # Set dealer's current sum
        if not card_is_ace or (card_is_ace and dealer.get_sum()+11 <= 21):
            dealer.set_sum(dealer.get_sum() + card_value)
        else:
            dealer.set_sum(dealer.get_sum() + 1)
        
        if dealer.get_sum() > 21:  # Dealer is BUSTED!
            return +1, player_history
    
    # End game, decides who wins.
    if player.get_sum() == dealer.get_sum():  # DRAW
        return 0, player_history
    elif player.get_sum() > dealer.get_sum():  # WIN
        return +1, player_history
    else:  # LOSE
        return -1, player_history

    
def mc_prediction(num_episodes, gamma=1.0):  # On-policy Monte Carlo Prediction
    assert num_episodes > 0, 'NUM_EPISODES CANNOT BE LESS THAN OR EQUAL 0.'

    usable_ace_total_rewards = np.zeros(shape=(10,10), dtype=np.float64)
    usable_ace_N = np.ones(shape=(10,10), dtype=np.int64)
    no_usable_ace_total_rewards = np.zeros(shape=(10,10), dtype=np.float64)
    no_usable_ace_N = np.ones(shape=(10,10), dtype=np.int64)
    
    # Init states
    dealer = BlackjackDealer()
    dealer.init_state()
    player = BlackjackPlayer()
    player.init_state(dealer)
    
    for episode in tqdm(range(num_episodes)):
        reward, player_history = simulation(player, dealer)
        for (player_sum, dealer_upcard, num_aces_player) in reversed(player_history):
            upcard_value = min(10, dealer_upcard)
            if num_aces_player != 0:  # Player has usable ace.
                usable_ace_total_rewards[player_sum-12][upcard_value-1] = gamma*usable_ace_total_rewards[player_sum-12][upcard_value-1] + reward
                usable_ace_N[player_sum-12][upcard_value-1] += 1
            else:  # Player has no usable ace.
                no_usable_ace_total_rewards[player_sum-12][upcard_value-1] = gamma*no_usable_ace_total_rewards[player_sum-12][upcard_value-1] + reward
                no_usable_ace_N[player_sum-12][upcard_value-1] += 1

        # Reset to init states.
        dealer.reset()
        dealer.init_state()
        player.reset()
        player.init_state(dealer)
    
    return usable_ace_total_rewards/usable_ace_N, no_usable_ace_total_rewards/no_usable_ace_N  # Sampling average

def visualize_figure_5_1(V_usable_ace_10k, V_no_usable_ace_10k, V_usable_ace_500k, V_no_usable_ace_500k, plot=False, save=False):
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.rc('font', size=6)

    ax = plt.subplot(2,2,1)
    plt.title('10k EPS WITH USABLE ACE')
    plt.xticks(ticks=np.arange(0,10,1), labels=np.arange(10,1-1,-1))
    plt.yticks(ticks=np.arange(0,10,1), labels=np.arange(12,21+1,1))
    plt.imshow(V_usable_ace_10k, cmap='viridis', interpolation='nearest')
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)

    plt.subplot(2,2,2, sharex=ax, sharey=ax)
    plt.title('500k EPS WITH USABLE ACE')
    plt.xticks(ticks=np.arange(0,10,1), labels=np.arange(10,1-1,-1))
    plt.yticks(ticks=np.arange(0,10,1), labels=np.arange(12,21+1,1))
    plt.imshow(V_usable_ace_500k, cmap='viridis', interpolation='nearest')
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)

    plt.subplot(2,2,3, sharex=ax, sharey=ax)
    plt.title('10k EPS WITHOUT USABLE ACE')
    plt.xticks(ticks=np.arange(0,10,1), labels=np.arange(10,1-1,-1))
    plt.yticks(ticks=np.arange(0,10,1), labels=np.arange(12,21+1,1))
    plt.imshow(V_no_usable_ace_10k, cmap='viridis', interpolation='nearest')
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    plt.xlabel("DEALER'S UPCARD")
    plt.ylabel("PLAYER'S SUM")

    plt.subplot(2,2,4, sharex=ax, sharey=ax)
    plt.title('500k EPS WITHOUT USABLE ACE')
    plt.xticks(ticks=np.arange(0,10,1), labels=np.arange(10,1-1,-1))
    plt.yticks(ticks=np.arange(0,10,1), labels=np.arange(12,21+1,1))
    heatmap = plt.imshow(V_no_usable_ace_500k, cmap='viridis', interpolation='nearest')
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    plt.colorbar(heatmap)

    if plot:
        plt.show()

    if save:
        plt.savefig('./5_1.png')
        plt.close()
        

""" FIGURE 5.1 """
V_usable_ace_10k, V_no_usable_ace_10k = mc_prediction(num_episodes=10000, gamma=1.0)
V_usable_ace_500k, V_no_usable_ace_500k = mc_prediction(num_episodes=500000, gamma=1.0)
visualize_figure_5_1(V_usable_ace_10k, V_no_usable_ace_10k, V_usable_ace_500k, V_no_usable_ace_500k, plot=True, save=False)
