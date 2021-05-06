#!/Users/huynguyen/miniforge3/envs/math/bin/python3


"""

    Name: HUY NGUYEN
    *BLACKJACK PROBLEM.
    *FIGURE 5.1 IN THE BOOK.
    *FINITE, UNDISCOUNTING PROBLEM.
    *Assume that cards are drawn from an infinite deck (WITH REPLACEMENT).
    *First-visit MC on-policy prediction, for estimating V with given INPUT POLICY.

"""


from Plot import visualize_figure_5_1
from Environment import BlackjackDealer, BlackjackPlayer, evaluate_card
from tqdm import tqdm
import numpy as np


def simulation(player=None, dealer=None):  # Monte Carlo Sampling, return final_reward, player_history
    assert player is not None and dealer is not None, 'INVALID INPUT, CANNOT RUN SIMULATION.'

    player_history = []  # List of tuple (player's sum, dealer's upcard, number of aces from player)

    while True:  # Player's turn
        # Add to history for sampling average (MC Sampling/MC Method)
        player_history.append((player.get_sum(), dealer.get_upcard(), player.has_usable_ace()))

        action = player.policy()  # Player's decision, 1 is HIT (continue), 0 is STICK (stop).
        if action == 0:
            break

        card = dealer.deal_card()  # Draw a card
        player.add_card(card)  # Add card to player's hand
        if player.get_sum() > 21:  # Player is BUSTED!
            return -1, player_history
        
    while True:  # Dealer's turn
        action = dealer.policy()  # Dealer's decision, 1 is HIT (continue), 0 is STICK (stop).
        if action == 0:
            break

        card = dealer.deal_card()  # Draw a card
        dealer.add_card(card)  # Add card to dealer's hand
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
    assert gamma > 0 and gamma <= 1, 'GAMMA NEEDS TO BE 0 < gamma <= 1'

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
        for (player_sum, dealer_upcard, has_usable_ace) in reversed(player_history):
            upcard_value = min(10, dealer_upcard)
            if has_usable_ace:  # Player has usable ace.
                usable_ace_total_rewards[player_sum-12, upcard_value-1] = gamma*usable_ace_total_rewards[player_sum-12, upcard_value-1] + reward
                usable_ace_N[player_sum-12, upcard_value-1] += 1
            else:  # Player has no usable ace.
                no_usable_ace_total_rewards[player_sum-12, upcard_value-1] = gamma*no_usable_ace_total_rewards[player_sum-12, upcard_value-1] + reward
                no_usable_ace_N[player_sum-12, upcard_value-1] += 1

        # Reset to init states.
        dealer.reset()
        dealer.init_state()
        player.reset()
        player.init_state(dealer)
    
    return usable_ace_total_rewards/usable_ace_N, no_usable_ace_total_rewards/no_usable_ace_N  # Sampling average
 

""" FIGURE 5.1 """
def figure_5_1(show=False, save=False):
    V_usable_ace_10k, V_no_usable_ace_10k = mc_prediction(num_episodes=10000, gamma=1.0)
    V_usable_ace_500k, V_no_usable_ace_500k = mc_prediction(num_episodes=500000, gamma=1.0)
    visualize_figure_5_1(V_usable_ace_10k, V_no_usable_ace_10k, V_usable_ace_500k, V_no_usable_ace_500k, show, save)

figure_5_1(show=True, save=False)
