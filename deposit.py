def deposit(accounts, account_number, amount):
    if amount > 0 and account_number in accounts:
        accounts[account_number]['balance'] += amount
        accounts[account_number]['transactions'].append(f"Deposit: +${amount}")
        print(f"Deposited ${amount}. New balance: ${accounts[account_number]['balance']}")
    elif amount <= 0:
        print("Invalid deposit amount. Please enter a positive value.")
    else:
        print(f"Account number {account_number} not found.")
