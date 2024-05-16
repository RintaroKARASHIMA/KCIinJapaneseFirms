# src/variable.py


## 初期条件
AR = 'app'  # app or reg
YEAR_STYLE = 'nendo'  # nendo or year

YEAR_START = 1981  # int
YEAR_END = 2010  # int
YEAR_RANGE = 10  # int

COUNTRY = 'JP'  # JP or EU

REGION_CORPORATION = 'right_person_name' # right_person_name or right_person_addr
CLASSIFICATION = 'schmoch35'  # ipc3 or ipc4 or schmoch
VALUE = 'reg_num'
CLASS_WEIGHT = 'fraction'  # fraction or duplication
APPLICANT_WEIGHT = 'fraction'  # fraction or duplication

# reg_num filter
EXTRACT_POPULATION = 'all'  # all or sep_year
TOP_P_OR_NUM = ('p', 3)  # (p or num, int)



COLOR_LIST = [
    'turquoise',
    'gold',
    'lime',
    'indigo',
    'red',
    'coral',
    'navy',
    'skyblue',
    'tomato',
    'olive',
    'cyan',
    'darkred',
    'darkgreen',
    'darkblue',
    'darkorange',
    'darkviolet',
    'deeppink',
    'firebrick',
    'darkcyan',
    'darkturquoise',
    'darkslategray',
    'darkgoldenrod',
    'mediumblue',
    'mediumseagreen',
    'mediumpurple',
    'mediumvioletred',
    'midnightblue',
    'saddlebrown',
    'seagreen',
    'sienna',
    'steelblue',
]
