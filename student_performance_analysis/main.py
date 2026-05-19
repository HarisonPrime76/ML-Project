from sources import data_loader

def main():
    data_path = "datasets/student_performance.csv"
    data = data_loader.load_data(data_path)

    if data :
        print('true')
    
    return 0
    