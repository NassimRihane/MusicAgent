import random

def update_motivation(p_m_old, p_event_given_M, p_event_given_notM):
    """
    Update P(M) given an event (success or failure) using Bayes' rule.
    p_event_given_M: P(event | M)
    p_event_given_notM: P(event | not M)
    """
    p_notM_old = 1 - p_m_old
    # Total probability of the event:
    p_event = p_event_given_M * p_m_old + p_event_given_notM * p_notM_old
    # Posterior probability P(M | event):
    p_m_new = (p_event_given_M * p_m_old) / p_event
    return p_m_new

# Initial probability of motivation
p_M = random.uniform(0.4,0.6)


# Define conditional probabilities for each interaction:

# Interaction 1 (Ice-breaker story)
p_success_M1 = random.uniform(0.55,0.65) #0.6
p_success_notM1 = random.uniform(0.25,0.35) #0.3
p_fail_M1 = 1 - p_success_M1
p_fail_notM1 = 1 - p_success_notM1

# Interaction 2 (Motivational analogy)
p_success_M2 = 0.7
p_success_notM2 = 0.2
p_fail_M2 = 1 - p_success_M2
p_fail_notM2 = 1 - p_success_notM2

# Interaction 3 (Challenge question)
p_success_M3 = 0.5
p_success_notM3 = 0.1
p_fail_M3 = 1 - p_success_M3
p_fail_notM3 = 1 - p_success_notM3


print(f"Initial P(Motivated) = {p_M:.2f}")
#for _ in range(5):
#    p_M = update_motivation



# Suppose we observe: Interaction1=success, Interaction2=fail, Interaction3=success
p_M = update_motivation(p_M, p_success_M1, p_success_notM1)
print(f"After Interaction1 (success): P(Motivated) = {p_M:.4f}")
p_M = update_motivation(p_M, p_fail_M2, p_fail_notM2)  # observed failure in Inter2
print(f"After Interaction2 (failure): P(Motivated) = {p_M:.4f}")
p_M = update_motivation(p_M, p_success_M3, p_success_notM3)
print(f"After Interaction3 (success): P(Motivated) = {p_M:.4f}")
