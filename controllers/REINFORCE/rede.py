
class Policy_Network(torch.nn.Module):
    """Neural network model representing the policy network."""
    
    def __init__(self, input_size, hidden_size, output_size):
        super(Policy_Network, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, hidden_size*2) 
        self.fc3 = nn.Linear(hidden_size*2, output_size)
    
    def forward(self, x):
        """Performs the forward pass through the network and computes action probabilities."""
        
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        
        return torch.softmax(x, dim=0)
    


class Agent_REINFORCE():
    """Agent implementing the REINFORCE algorithm."""

    def __init__(self, save_path, load_path, num_episodes, max_steps, 
                  learning_rate, gamma, hidden_size, clip_grad_norm, baseline):
                
        self.save_path = save_path
        self.load_path = load_path
        
        os.makedirs(self.save_path, exist_ok=True)
        
        # Hyper-parameters Attributes
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.learing_rate = learning_rate
        self.gamma = gamma
        self.hidden_size = hidden_size
        self.clip_grad_norm = clip_grad_norm
        self.baseline = baseline
        
        # Initialize Network (Model)
        self.network = Policy_Network(input_size=3, hidden_size=self.hidden_size, output_size=3).to(device)
    
        # Create the self.optimizers
        self.optimizer = optim.Adam(self.network.parameters(), self.learing_rate)
        
        # instance of env
        self.env = Environment()
               
        
    def save(self, path):
        """Save the trained model parameters after final episode and after receiving the best reward."""
        torch.save(self.network.state_dict(), self.save_path + path)
    
    
    def load(self):
        """Load pre-trained model parameters."""
        self.network.load_state_dict(torch.load(self.load_path, map_location=torch.device('cpu')))


    def compute_returns(self, rewards):
        """
        Compute the discounted returns.
        
        Parameters:
        - rewards (list): List of rewards obtained during an episode.
        
        Returns:
        - torch.Tensor: Computed returns.
        """

        # Generate time steps and calculate discount factors
        t_steps = torch.arange(len(rewards))
        discount_factors = torch.pow(self.gamma, t_steps).to(device)
    
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    
        # Calculate returns using discounted sum
        returns = rewards * discount_factors
        returns = returns.flip(dims=(0,)).cumsum(dim=0).flip(dims=(0,)) / discount_factors
    
        if self.baseline:
            mean_reward = torch.mean(rewards)
            returns -= mean_reward
        
        return returns

    
    def compute_loss(self, log_probs, returns):
        """
        Compute the REINFORCE loss.
        
        Parameters:
        - log_probs (list): List of log probabilities of actions taken during an episode.
        - returns (torch.Tensor): Computed returns for the episode.
        
        Returns:
        - torch.Tensor: Computed loss.
        """
            
        # Calculate loss for each time step
        loss = []
        for log_prob, G in zip(log_probs, returns):
            loss.append(-log_prob * G)
            
        # Sum the individual losses to get the total loss
        return torch.stack(loss).sum()
    
    
    def train(self):
        """
        Train the agent using the REINFORCE algorithm.
        
        This method performs the training of the agent using the REINFORCE algorithm. It iterates
        over episodes, collects experiences, computes returns, and updates the policy network.
        """           
        
        self.network.train()
        start_time = time.time()
        reward_history = []
        best_score = -np.inf
        for episode in range(1, self.num_episodes+1):
            done = False
            state = self.env.reset()
                
            log_probs = []
            rewards = []
            ep_reward = 0
            while True:
                action_probs = self.network(torch.as_tensor(state, device=device)) # action probabilities
                dist = torch.distributions.Categorical(action_probs) # Make categorical distrubation
                action = dist.sample() # Sample action
                log_prob = dist.log_prob(action) # The log probability of the action under the current policy distribution.
                log_probs.append(log_prob)
                next_state, reward, done = self.env.step(action.item(), self.max_steps)
                
                rewards.append(reward)
                ep_reward += reward

                if done:
                    returns = self.compute_returns(rewards)
                    loss = self.compute_loss(log_probs, returns)
                    self.optimizer.zero_grad()
                    loss.backward()
                    # grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(self.network.parameters(), float('inf'))
                    # print("Gradient norm before clipping:", grad_norm_before_clip)
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.clip_grad_norm)
                    self.optimizer.step()
                    reward_history.append(ep_reward)
                              
                    if ep_reward > best_score:
                        self.save(path='/best_weights.pt')
                        best_score = ep_reward
                    
                    print(f"Episode {episode}: Score = {ep_reward:.3f}")
                    break
                
                state = next_state
         
        # Save final weights and plot reward history
        self.save(path='/final_weights.pt')
        self.plot_rewards(reward_history)        
                
        # Print total training time
        elapsed_time = time.time() - start_time
        elapsed_timedelta = timedelta(seconds=elapsed_time)
        formatted_time = str(elapsed_timedelta).split('.')[0]
        print(f'Total Spent Time: {formatted_time}')
        
              
    def test(self):
        """
        Test the trained agent.
        This method evaluates the performance of the trained agent.
        """
        
        start_time = time.time()
        rewards = []
        self.load()
        self.network.eval()
        
        for episode in range(1, self.num_episodes+1):
            state = self.env.reset()
            done = False
            ep_reward = 0
            while not done:
                action_probs = self.network(torch.as_tensor(state, device=device))
                action = torch.argmax(action_probs, dim=0)
                state, reward, done = self.env.step(action.item(), self.max_steps)
                ep_reward += reward
            rewards.append(ep_reward)
            print(f"Episode {episode}: Score = {ep_reward:.3f}")
        print(f"Mean Score = {np.mean(rewards):.3f}")
        
        elapsed_time = time.time() - start_time
        elapsed_timedelta = timedelta(seconds=elapsed_time)
        formatted_time = str(elapsed_timedelta).split('.')[0]
        print(f'Total Spent Time: {formatted_time}')
    
    
    def plot_rewards(self, rewards):
        # Calculate the Simple Moving Average (SMA) with a window size of 25
        sma = np.convolve(rewards, np.ones(25)/25, mode='valid')
        
        plt.figure()
        plt.title("Episode Rewards")
        plt.plot(rewards, label='Raw Reward', color='#142475', alpha=0.45)
        plt.plot(sma, label='SMA 25', color='#f0c52b')
        plt.xlabel("Episode")
        plt.ylabel("Rewards")
        plt.legend()
        
        plt.savefig(self.save_path + '/reward_plot.png', format='png', dpi=1000, bbox_inches='tight')
        plt.tight_layout()
        plt.grid(True)
        plt.show()
        plt.clf()
        plt.close()
            
