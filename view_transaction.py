def view_transactions(accounts, account_number):
    if account_number in accounts:
        print(f"Transaction history for {accounts[account_number]['holder_name']}'s account:")
        for transaction in accounts[account_number]['transactions']:
            print(transaction)
    else:
        print(f"Account number {account_number} not found.")
