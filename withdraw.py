def withdraw(accounts, account_number, amount):
    if 0 < amount <= accounts[account_number]['balance'] and account_number in accounts:
        accounts[account_number]['balance'] -= amount
        accounts[account_number]['transactions'].append(f"Withdrawal: -${amount}")
        print(f"Withdrew ${amount}. New balance: ${accounts[account_number]['balance']}")
    elif amount <= 0:
        print("Invalid withdrawal amount. Please enter a positive value.")
    else:
        print("Invalid withdrawal amount. Insufficient funds.")
