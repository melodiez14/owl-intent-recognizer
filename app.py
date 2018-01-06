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
    import evaluate as ev
    ev.evaluate()

def train():
    '''Train the model for the production ready and no splitting data into train and test'''
    import train as tr
    tr.train()

def run():
    '''Run the application'''
    print('application run')

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
