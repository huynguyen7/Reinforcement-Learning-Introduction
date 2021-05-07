#!/Users/huynguyen/miniforge3/envs/math/bin/python3


"""

    Name: HUY NGUYEN
    *BLACKJACK PROBLEM.
    *FIGURE 5.2 IN THE BOOK.
    *FINITE, UNDISCOUNTING PROBLEM.
    *Assume that cards are drawn from an infinite deck (WITH REPLACEMENT).
    *First-Visit MC Control With Exploring Starts(Policy Prediction + Improvement), for estimating OPTIMAL POLICIES and OPTIMAL STATE VALUES.
    *Just another version (WORSE, but follow the book's pseudocode) with stochastic policy init, then update Q and Pi within each time step per episode.

"""


from Plot import visualize_figure_5_2
from Environment import BlackjackDealer, BlackjackPlayer, evaluate_card
from tqdm import tqdm
import numpy as np


def simulation(player=None, dealer=None, init_action=None, usable_ace_optimal_pi=None, no_usable_ace_optimal_pi=None):  # Monte Carlo Sampling, return final_reward, player_history
    assert player is not None and dealer is not None and usable_ace_optimal_pi is not None and no_usable_ace_optimal_pi is not None, 'INVALID INPUT, CANNOT RUN SIMULATION.'

    player_history = []  # List of tuple (player's sum, dealer's upcard, player has usable ace)

    while True:  # Player's turn
        if init_action is not None:
            action = init_action
            init_action = None
        elif player.has_usable_ace():
            action = usable_ace_optimal_pi[player.get_sum()-12, min(10, dealer.get_upcard())-1]
        else:
            action = no_usable_ace_optimal_pi[player.get_sum()-12, min(10, dealer.get_upcard())-1]

        # Add to history for sampling average (MC Sampling/MC Method)
        player_history.append(((player.get_sum(), dealer.get_upcard(), player.has_usable_ace()), action))

        # POLICY: Player's decision, 1 is HIT (continue), 0 is STICK (stop).
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


def mc_epsilon_greedy(num_episodes=500000, gamma=1.0, epsilon=0.1):  # Monte Carlo Exploring Starts
    assert num_episodes > 0, 'NUM_EPISODES CANNOT BE LESS THAN OR EQUAL 0.'
    assert gamma > 0 and gamma <= 1, 'GAMMA NEEDS TO BE 0 < gamma <= 1'

    # state ~ (player_sum, dealer_upcard), action ~ action
    # => Q(s,a) ~ Q((player_sum, dealer_upcard), action)

    
    usable_ace_optimal_pi = np.random.randint(low=0,high=2,size=(10,10),dtype=np.int8)  # Stochastic policies
    no_usable_ace_optimal_pi = np.random.randint(low=0,high=2,size=(10,10),dtype=np.int8)  # Stochastic policies
    
    usable_ace_Q = np.zeros(shape=(10,10,2), dtype=np.float64)
    no_usable_ace_Q = np.zeros(shape=(10,10,2), dtype=np.float64)

    usable_ace_returns = np.zeros(shape=(10,10,2), dtype=np.float64)  # player_sum, dealer_upcard, action 
    usable_ace_N = np.ones(shape=(10,10,2), dtype=np.int64)
    no_usable_ace_returns = np.zeros(shape=(10,10,2), dtype=np.float64)  # player_sum, dealer_upcard, action
    no_usable_ace_N = np.ones(shape=(10,10,2), dtype=np.int64)

    dealer = BlackjackDealer()
    player = BlackjackPlayer()
    visited_states = set()

    for episode in tqdm(range(num_episodes)):
        # Stochastic initialization.
        dealer.reset()
        dealer.init_state()
        player.reset()
        player.init_state(dealer)
        init_action = np.random.randint(0,2)

        reward, player_history = simulation(player, dealer, init_action, usable_ace_optimal_pi, no_usable_ace_optimal_pi)
        
        G = 0  # Return
        visited_states.clear()  # Flush out all (state, action) pairs from previous episode.
        for ((player_sum, dealer_upcard, has_usable_ace), action) in reversed(player_history):
            dealer_upcard = min(10, dealer_upcard)
            # If (state,action) pair is already visited, no need to update => FIRST VISIT UPDATE.
            state_action = ((player_sum, dealer_upcard, has_usable_ace), action)
            if state_action not in visited_states:
                visited_states.add(state_action)
                if has_usable_ace:
                    usable_ace_returns[player_sum-12, dealer_upcard-1, action] = gamma*usable_ace_returns[player_sum-12, dealer_upcard-1, action] + reward
                    usable_ace_N[player_sum-12, dealer_upcard-1, action] += 1
                    usable_ace_Q[player_sum-12, dealer_upcard-1, action] = usable_ace_returns[player_sum-12, dealer_upcard-1, action]/usable_ace_N[player_sum-12, dealer_upcard-1, action]  # Sampling average
                    state_action_values = usable_ace_Q[player_sum-12, dealer_upcard-1, :]
                    usable_ace_optimal_pi[player_sum-12, dealer_upcard-1] = np.random.choice([action for action, value in enumerate(state_action_values) if value == np.max(state_action_values)])  # List comprehension with predicate.

                else:
                    no_usable_ace_returns[player_sum-12, dealer_upcard-1, action] = gamma*no_usable_ace_returns[player_sum-12, dealer_upcard-1, action] + reward
                    no_usable_ace_N[player_sum-12, dealer_upcard-1, action] += 1
                    no_usable_ace_Q[player_sum-12, dealer_upcard-1, action] = no_usable_ace_returns[player_sum-12, dealer_upcard-1, action]/no_usable_ace_N[player_sum-12, dealer_upcard-1, action]  # Sampling average
                    state_action_values = no_usable_ace_Q[player_sum-12, dealer_upcard-1, :]
                    no_usable_ace_optimal_pi[player_sum-12, dealer_upcard-1] = np.random.choice([action for action, value in enumerate(state_action_values) if value == np.max(state_action_values)])  # List comprehension with predicate.

    return usable_ace_Q, no_usable_ace_Q, usable_ace_optimal_pi, no_usable_ace_optimal_pi

def figure_5_2(num_episodes=500000, show=False, save=False):
    usable_ace_Q, no_usable_ace_Q, usable_ace_optimal_pi, no_usable_ace_optimal_pi = mc_epsilon_greedy(num_episodes, gamma=1.0, epsilon=0.1)
    # Find optimal state values.
    usable_ace_optimal_V = np.max(usable_ace_Q, axis=-1)
    no_usable_ace_optimal_V = np.max(no_usable_ace_Q, axis=-1)
    visualize_figure_5_2(usable_ace_optimal_V, usable_ace_optimal_pi, no_usable_ace_optimal_V, no_usable_ace_optimal_pi, show, save)

figure_5_2(show=True, save=False)
