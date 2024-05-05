import re
import matplotlib.pyplot as plt

class TrainingData:
    def __init__(self, filename, label):
        self.file_path = f"training_data/{filename}.txt"
        self.label = label 
        self.steps = []
        self.rewards = []
        self.policies = []
        self.values = []
        self.entropies = []
        self.confidences = []
        self.scores = []
        self.vocabs = []
        self._parse_file()

    def _parse_file(self):
        """Reads file and parses the training data into lists."""
        with open(self.file_path, 'r') as file:
            for line in file:
                if "reward:" in line:
                    self.steps.append(int(re.search(r'(\d+). reward:', line).group(1)))
                    self.rewards.append(float(re.search(r'reward:\s+([-\d.]+)', line).group(1)))
                    self.policies.append(float(re.search(r'policy:\s+([-\d.]+)', line).group(1)))
                    self.values.append(float(re.search(r'value:\s+([-\d.]+)', line).group(1)))
                    self.entropies.append(float(re.search(r'entropy:\s+([-\d.]+)', line).group(1)))
                    self.confidences.append(float(re.search(r'confidence:\s+([-\d.]+)', line).group(1)))
                    self.scores.append(int(re.search(r'score:\s+(\d+)', line).group(1)))
                    self.vocabs.append(int(re.search(r'vocab:\s+(\d+)', line).group(1)))

    def get_data_as_dict(self):
        """Returns the parsed data as a dictionary."""
        return {
            "steps": self.steps,
            "rewards": self.rewards,
            "policies": self.policies,
            "values": self.values,
            "entropies": self.entropies,
            "confidences": self.confidences,
            "scores": self.scores,
            "vocabs": self.vocabs
        }
    
    def plot_reward_over_time(self):
        """Plots the reward values over the training steps."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.steps, self.rewards, marker='o', linestyle='-', color='blue')
        plt.title('Reward Over Time')
        plt.xlabel('Training Steps')
        plt.ylabel('Reward')
        plt.grid(True)
        plt.show()

    def plot_policy_value_loss_over_time(self):
        """Plots both policy and value losses over the training steps."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.steps, self.policies, marker='o', linestyle='-', color='red', label='Policy Loss')
        plt.plot(self.steps, self.values, marker='o', linestyle='-', color='green', label='Value Loss')
        plt.title('Policy and Value Loss Over Time')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_entropy_confidence_over_time(self):
        """Plots entropy and confidence over the training steps."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.steps, self.entropies, marker='o', linestyle='-', color='purple', label='Entropy')
        plt.plot(self.steps, self.confidences, marker='o', linestyle='-', color='orange', label='Confidence')
        plt.title('Entropy and Confidence Over Time')
        plt.xlabel('Training Steps')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_rewards_comparison(*agents):
        """Plots rewards over time for multiple TrainingData instances."""
        plt.figure(figsize=(10, 6))
        for agent in agents:
            plt.plot(agent.steps, agent.rewards, label=agent.label)
        plt.title('Reward Comparison Over Time')
        plt.xlabel('Training Steps')
        plt.ylabel('Rewards')
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_losses_comparison(*agents):
        """Plots policy and value losses over time for multiple TrainingData instances."""
        fig, axs = plt.subplots(2, 1, figsize=(10, 12))
        for agent in agents:
            axs[0].plot(agent.steps, agent.policies, label=f'{agent.label} Policy')
            axs[1].plot(agent.steps, agent.values, label=f'{agent.label} Value')
        
        axs[0].set_title('Policy Loss Comparison Over Time')
        axs[0].set_xlabel('Training Steps')
        axs[0].set_ylabel('Policy Loss')
        axs[0].legend()
        axs[0].grid(True)
        
        axs[1].set_title('Value Loss Comparison Over Time')
        axs[1].set_xlabel('Training Steps')
        axs[1].set_ylabel('Value Loss')
        axs[1].legend()
        axs[1].grid(True)
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # dense_agent = TrainingData('training_a2c_multiple_5_dense', 'Dense')
    # balanced_agent = TrainingData('training_a2c_multiple_5_balanced', 'Balanced')
    # sparse_agent = TrainingData('training_a2c_multiple_5_sparse', 'Sparse')
    # TrainingData.plot_rewards_comparison(dense_agent, balanced_agent, sparse_agent)
    # TrainingData.plot_losses_comparison(dense_agent, balanced_agent, sparse_agent)

    all_agent = TrainingData('training_a2c_multiple_5_all', 'All')
    all_agent.plot_reward_over_time()
    all_agent.plot_policy_value_loss_over_time()
    all_agent.plot_entropy_confidence_over_time()