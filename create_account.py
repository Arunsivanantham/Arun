def create_account(accounts, account_number, holder_name, initial_balance=0):
    if account_number not in accounts:
        accounts[account_number] = {'holder_name': holder_name, 'balance': initial_balance, 'transactions': []}
        print(f"Account created for {holder_name} with account number {account_number}.")
    else:
        print(f"Account number {account_number} already exists.")
