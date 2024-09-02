"""
Name: Gianluca Bollo
EID: gb25625
C S 313E
Dr. Teymourian (Section 50775)
Term Project: 
OLS Linear Regression implementation using a linked list. 
Model analysis using a heap.
"""
import sys
import heapq

''' Class to create data point nodes '''
class DataPoint():
    ''' Constructor for Data point Nodes '''
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.exp_y = None
        self.residual = None
        self.next = None
    
    ''' Str method to print ordered pair of data point'''
    def __str__(self):
        return f'{self.x, self.y}'

''' Class to create a linked-list capable of storing csv file data '''
class LinkedListDataset():
    ''' Constructor for linked list storing data set '''
    def __init__(self):
        self.head = None
        self.tail = None
        self.ind_name = ''
        self.dep_name = ''
        self.pred_list = []
    
    def append_data_point(self, data_point):
        ''' Method to append data to linked list '''
        if self.head is None:
            self.head = data_point
            self.tail = data_point
        else:
            self.tail.next = data_point
            self.tail = data_point
    
    def get_length(self):
        ''' Method to retrieve and return length of linked list '''
        if self.head is None:
            return 0
        size = 0
        cur_node = self.head
        while cur_node is not None:
            size += 1
            cur_node = cur_node.next
        return size 
    
    def print_dataset(self):
        ''' Method to print dataset '''
        if self.head is None:
            print('Error: List is empty')
        else:
            print(f'X: {self.ind_name} Y: {self.dep_name}')
            cur_node = self.head
            while cur_node is not None:
                print(cur_node)
                cur_node = cur_node.next  

''' Class to create Linear Regression model '''
class LinearRegression():
    ''' Constructor method to initialize data and important stats '''
    def __init__(self, linked_list):
        self.dataset = linked_list
        self.xmean = None
        self.ymean = None
        self.slope = None
        self.intercept = None
        self.rsq = None
        self.stats = {}
        self.abs_residual_heap = []
    
    def calc_slope(self):
        ''' Method to calulcate and return slope of model using OLS '''
        cur_node = self.dataset.head
        xsum = 0
        ysum = 0
        n = self.dataset.get_length()
        while cur_node is not None:
            xsum += cur_node.x
            ysum += cur_node.y
            cur_node = cur_node.next
        self.xmean = xsum / n
        self.ymean = ysum / n
        cur_node = self.dataset.head
        numerator = 0
        denominator = 0
        while cur_node is not None:
            numerator += (cur_node.x - self.xmean) * (cur_node.y - self.ymean)
            denominator += (cur_node.x - self.xmean) ** 2
            cur_node = cur_node.next
        self.slope = numerator / denominator
        return self.slope
   
    def calc_intercept(self):
        ''' Method to calulcate and return intercept of model '''
        self.intercept = self.ymean - (self.slope * self.xmean)
        return self.intercept
    
    def set_expected_data(self):
        ''' 
        Method to update expected value and residual of each data point
        in the dataset based on regression equation. Pushes all 
        residuals onto a heap.
        '''
        cur_node = self.dataset.head
        while cur_node is not None:
            cur_node.exp_y = (self.slope * cur_node.x) + self.intercept
            cur_node.residual = cur_node.y - cur_node.exp_y
            cur_node = cur_node.next

    def calc_rsq(self):
        ''' Method to calculate and return r^2 value of model '''
        cur_node = self.dataset.head
        ssr = 0
        sst = 0
        while cur_node is not None:
            ssr += (cur_node.exp_y - self.ymean) ** 2
            sst += (cur_node.y - self.ymean) ** 2
            cur_node = cur_node.next
        rsq = ssr / sst
        return rsq
        
    def print_stats(self):
        ''' 
        Method to print stats of model (slope, intercept, and r^2
        '''
        self.stats['Slope'] = round(self.calc_slope(), 3)
        self.stats['Intercept'] = round(self.calc_intercept(), 3)
        self.set_expected_data()
        self.stats['R-Squared'] = round(self.calc_rsq(), 3)
        print(f'Linear Regression Model for {self.dataset.ind_name} vs {self.dataset.dep_name}:')
        for key, value in self.stats.items():
            print(f'{key}: {value}')
    
    def k_residuals(self):
        ''' 
        This is a form of model analysis, displaying the model's
        k biggest and smallest misses based on data file length / 4.
        '''
        k = self.dataset.get_length() // 25
        cur_node = self.dataset.head
        while cur_node is not None:
            heapq.heappush(self.abs_residual_heap, round(abs(cur_node.residual), 3))
            cur_node = cur_node.next
        print(f'Largest {k} misses of the model: {heapq.nlargest(k, self.abs_residual_heap)}')
        print(f'Smallest {k} misses of the model: {heapq.nsmallest(k, self.abs_residual_heap)}')
    
    def predict(self):
        ''' 
        Method to make prediction for an input based on model
        Returns prediction
        '''
        print('Predicted Values from Model:')
        for given_x in self.dataset.pred_list:
            print(f'Predicted {self.dataset.dep_name} value for {self.dataset.ind_name} {given_x}: {round((self.slope * given_x) + self.intercept, 3)}')

def load_data(file):
    ''' 
    Function to populate linked list with a given csv file 
    Returns populated linked list
    '''
    data_list = LinkedListDataset()
    lines = file.readlines()
    label_line = lines.pop(0)
    label_line = label_line.strip()
    var_labels = label_line.split(',')
    data_list.ind_name = var_labels[0]
    data_list.dep_name = var_labels[1]
    for line in lines:
        nums = line.strip()
        nums = line.split(',')
        if nums[1] == 'p\n': 
            data_list.pred_list.append(float(nums[0]))
        else:
            readin_x = float(nums[0])
            readin_y = float(nums[1])
            data_point = DataPoint(readin_x, readin_y)
            data_list.append_data_point(data_point)
    return data_list

def main():
    ''' Main function to print model results and analysis '''
    data = load_data(sys.stdin)
    model = LinearRegression(data)
    model.print_stats()
    model.predict()
    model.k_residuals()

    ### set backs - assuming linear relationship, correct format of input dataset

if __name__ == '__main__':
    main()