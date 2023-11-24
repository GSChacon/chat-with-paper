class HandleUserInput:
    def __init__(self):
        pass

    def ask_user_input(self, first_message = True):
        if first_message:
            user_input = input('Ask questions about your paper (type END to exit): \n')
        else:
            user_input = input('\n')
        return user_input

    def check_end(self, user_input):
        if user_input == 'END':
            return True
        return False
