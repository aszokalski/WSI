from pomegranate import *
from notebooks.lab7.src.probabilities import *

# root nodes
nigerian_keyword = DiscreteDistribution(nigerian_probs)
prince_keyword = DiscreteDistribution(prince_probs)
send_keyword = DiscreteDistribution(send_probs)
domain = DiscreteDistribution(domain_probs)

# non-root nodes
is_spam = ConditionalProbabilityTable(
    is_spam_cpt_entries, [nigerian_keyword, prince_keyword, send_keyword, domain]
)

s1 = State(nigerian_keyword, name="nigerian_keyword")
s2 = State(prince_keyword, name="prince_keyword")
s3 = State(send_keyword, name="send_keyword")
s4 = State(domain, name="domain")
s5 = State(is_spam, name="is_spam")

network = BayesianNetwork("Spam Filter")

network.add_states(s1, s2, s3, s4, s5)
network.add_edge(s1, s5)
network.add_edge(s2, s5)
network.add_edge(s3, s5)
network.add_edge(s4, s5)

network.bake()

if __name__ == "__main__":
    print(
        network.predict_proba(
            {
                "nigerian_keyword": "nigerian_keyword_yes",
                "prince_keyword": "prince_keyword_yes",
                "send_keyword": "send_keyword_yes",
            }
        )
    )
