import sys
import csv
import pickle
import regex


# Person class, stores information about an employee, comes with display() method
#   to display all relevant fields about an employee
class Person:
    def __init__(self, last: str, first: str, mi: str, id: str, phone: str):
        self.last = last
        self.first = first
        self.mi = mi
        self.id = id
        self.phone = phone

    def display(self):
        print(f'Employee id: {self.id}')
        print(f'\t\t{self.first} {self.mi} {self.last}')
        print(f'\t\t{self.phone}')


# Function that reads a given file path for a csv file with information about each employee
#   Parses the file and then returns a dictionary with People objects representing each entry
def get_people(file_path):
    people = {}
    with open(str(file_path)) as f:
        for line in list(csv.reader(f))[1:]:
            line[:2] = [s.capitalize() for s in line[:2]]
            # middle initial invalid error
            if len(line[2]) > 1:
                print(f'ERROR: Entry for employee middle initial is more than 1 letter')
                return
            line[2] = line[2].upper() if line[2] else 'X'
            # employee ID invalid error
            while not regex.match("^[A-Z]{2}[0-9]{4}$", line[3]):
                print(f'Employee ID: {line[3]} for {line[1]} {line[0]} invalid format, please re-enter valid ID'
                      f'(2 capital letters then 4 numbers')
                line[3] = input()
            number = [c for c in line[4] if c.isnumeric()]
            # Employee phone number not 10 digits error
            #   (if the phone number is improperly formatted but still has 10 digits
            #    then this will automatically format the number)
            while len(number) != 10:
                print(f'Employee phone number: {line[4]} for {line[1]} {line[0]} has too many digits: {len(number)}'
                      f' digits\nPlease re-enter')
                line[4] = input()
                number = [c for c in line[4] if c.isnumeric()]
            line[4] = ''.join(number[:3]) + '-' + ''.join(number[3:6]) + '-' + ''.join(number[6:])
            # print(line)
            people[line[3]] = Person(line[0], line[1], line[2], line[3], line[4])
    return people


# Main method, checks if file path is given as arg, else asks again
def main():
    if len(sys.argv) < 2:
        print('Relative file path not given in args. Exiting...')
        return
    people = get_people(sys.argv[1])

    # Saving people dict to pickle file
    pickle.dump(people, open('dict.p', 'wb'))

    # Using pickle file to display employee information
    people_pickle = pickle.load(open('dict.p', 'rb'))
    print()
    print(f'Employee list:\n')
    for person in people_pickle.values():
        person.display()
        print()


if __name__ == "__main__":
    main()
