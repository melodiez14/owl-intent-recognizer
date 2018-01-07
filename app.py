'''sys used for read the argument typed by users'''
import sys

def welcome():
    '''This function used for printing the usage of this application'''

    print('''\
Welcome to owl assistant bot wizard
Usage:
    python app.py [arguments]
The commands are:
    evaluate
    train
    run
    version
    help
    ''')

def version():
    '''Show the application version'''
    print('Owl assistant v1')

def evaluate():
    '''Evaluate all of built CNN model and visualize the accuracy, precision, and recall'''
    from evaluate import Evaluator
    from model import Model1, Model2, Model3, Model4, Model5, Model6
    evaluator = Evaluator(Model1, Model2, Model3, Model4, Model5, Model6)
    evaluator.evaluate()

def train_print():
    '''Print the helper of training subprogram'''
    print('''\
Please specified the model
    one\t\tNR, CF, TKN, Conv1D, MaxPooling
    two\t\tNR, CF, TKN, SWR, Conv1D, MaxPooling
    three\tNR, CF, TKN, SWR, STM, Conv1D, MaxPooling
    four\tNR, CF, TKN, Conv1D, MaxPooling, Conv1D, MaxPooling
    five\tNR, CF, TKN, SWR, Conv1D, MaxPooling, Conv1D, MaxPooling
    six\t\tNR, CF, TKN, SWR, STM, Conv1D, MaxPooling, Conv1D, MaxPooling
Examples:
    python app.py train one
''')
    exit()

def train():
    '''Train the model for the production ready and no splitting data into train and test'''
    if len(sys.argv) < 3:
        train_print()
    from model import Model1, Model2, Model3, Model4, Model5, Model6
    from evaluate import Trainner
    model = {
        'one': Model1,
        'two': Model2,
        'three': Model3,
        'four': Model4,
        'five': Model5,
        'six': Model6
    }.get(sys.argv[2], train_print)
    trainner = Trainner(model)
    trainner.train()

def run():
    '''Run the application'''
    from evaluate import Webserver
    server = Webserver()
    server.run()

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        welcome()
        exit()
    INTENT = {
        'evaluate': evaluate,
        'train': train,
        'run': run,
        'version': version,
        'help': welcome,
    }.get(sys.argv[1], welcome)
    INTENT()
