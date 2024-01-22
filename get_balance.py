def get_balance(accounts, account_number):
    if account_number in accounts:
        return accounts[account_number]['balance']
    else:
        print(f"Account number {account_number} not found.")
        return None
