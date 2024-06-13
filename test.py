import pandas as pd
import numpy as np

# Define the number of samples
num_samples = 100

# Create a DataFrame with random values
df = pd.DataFrame({ 'totaldue': np.random.randint(0, 2, num_samples), 'education_flag': np.random.randint(0, 2, num_samples), 'employment_flag': np.random.randint(0, 2, num_samples),
    'bnk_acc_typ_flag': np.random.randint(0, 2, num_samples),
    'loan_freq': np.random.randint(1, 10, num_samples),
    'avg_paybacktime': np.random.rand(num_samples),
    'clients_avg_loanAMT': np.random.rand(num_samples),
    'clients_max_loanAMT': np.random.rand(num_samples),
    'total_loans': np.random.randint(1, 10, num_samples),
    'ontime_loans': np.random.randint(1, 10, num_samples),
    'ontime_percentage': np.random.rand(num_samples),
    'pays_late_risk': np.random.randint(0, 2, num_samples),
    'referred_client': np.random.randint(0, 2, num_samples),
    'loan_interest2': np.random.rand(num_samples),
    '%loan_int2': np.random.rand(num_samples),
    'age_categ_youth': np.random.randint(0, 2, num_samples),
    'age_categ_young_adult': np.random.randint(0, 2, num_samples),
    'age_categ_adult': np.random.randint(0, 2, num_samples),
    'age_categ_retired': np.random.randint(0, 2, num_samples),
    'BorrowerReliabilityScore': np.random.rand(num_samples)
})

# Save the DataFrame to a CSV file
df.to_csv('dummy_data.csv', index=False)