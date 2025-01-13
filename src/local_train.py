from train import ProjectAgent

if __name__ == "__main__":
    agent = ProjectAgent()

    print("Training the agent with FQI...")
    agent.train(nb_iter=100)

    agent.save("trained_agent.pkl")
