import numpy as np


def evaluate_card(card):  # Evaluate card and return tuple (card_is_ace, card_value).
    return (False, min(10, card)) if card != 1 else (True, 1)


class PlayerFrame(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.cards = []
        self.usable_ace = False

    def get_sum(self):
        num_aces = 0
        hand_sum = 0
        for card in self.cards:
            (card_is_ace, card_value) = evaluate_card(card)
            num_aces += 1 if card_is_ace else 0
            if not card_is_ace:
                hand_sum += card_value

        while num_aces > 1:
            num_aces -= 1
            hand_sum += 1

        if num_aces == 0:
            self.usable_ace = False
        elif num_aces == 1 and hand_sum+11 <= 21:
            hand_sum += 11
            self.usable_ace = True
        else:
            hand_sum += 1
            self.usable_ace = False

        return hand_sum
    
    def has_usable_ace(self):
        return self.usable_ace

    def get_num_aces(self):
        return self.num_aces

    def add_card(self, card):
        self.cards.append(card)
        
    def get_card_list(self):
        return self.cards


class BlackjackDealer(PlayerFrame):
    def policy(self):
        hand_sum = self.get_sum()
        return 1 if hand_sum >= 12 and hand_sum < 17 else 0

    def deal_card(self):  # Get a random card from uniform dist WITH REPLACEMENT
        return np.random.randint(low=1, high=14)

    def get_upcard(self):
        return self.cards[0]  # First card is upcard

    def init_state(self):  # Used for MC Prediction
        while len(self.cards) < 2:  # Dealer needs to have at least 2 cards at init state.
            card = self.deal_card()  # Draw a card
            self.add_card(card)


class BlackjackPlayer(PlayerFrame):
    def policy(self):
        hand_sum = self.get_sum()
        return 1 if hand_sum >= 12 and hand_sum < 20 else 0

    def init_state(self, dealer):
        while self.get_sum() < 12:  # Player needs to have at least cumulative sum of 12.
            card = dealer.deal_card()  # Draw a card
            self.add_card(card)

    def greedy_policy(self, dealer_upcard, usable_ace_returns, usable_ace_N, no_usable_ace_returns, no_usable_ace_N):
        player_sum = self.get_sum()
        dealer_upcard = min(10, dealer_upcard)

        # Get argmax of the average returns(s, a)
        if self.has_usable_ace():
            values = usable_ace_returns[player_sum-12, dealer_upcard-1, :]/usable_ace_N[player_sum-12, dealer_upcard-1, :]
        else:
            values = no_usable_ace_returns[player_sum-12, dealer_upcard-1, :]/no_usable_ace_N[player_sum-12, dealer_upcard-1, :]
        return np.random.choice([action for action, value in enumerate(values) if value == np.max(values)])  # List comprehension with predicate.
 
    def epsilon_greedy_policy(self, epsilon, dealer_upcard, usable_ace_returns, usable_ace_N, no_usable_ace_returns, no_usable_ace_N):
        player_sum = self.get_sum()
        dealer_upcard = min(10, dealer_upcard)

        greed = np.random.rand()  # Uniform dist
        if greed <= epsilon:
            return np.random.choice([0,1])
        else:
            # Get argmax of the average returns(s, a)
            if self.has_usable_ace():
                values = usable_ace_returns[player_sum-12, dealer_upcard-1, :]/usable_ace_N[player_sum-12, dealer_upcard-1, :]
            else:
                values = no_usable_ace_returns[player_sum-12, dealer_upcard-1, :]/no_usable_ace_N[player_sum-12, dealer_upcard-1, :]
            return np.random.choice([action for action, value in enumerate(values) if value == np.max(values)])  # List comprehension with predicate.

