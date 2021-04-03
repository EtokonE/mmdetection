# Определяем данные для обучения и валидации
from glob import glob 

# Корневая директория
data_root = '/home/user/Documents/Kalinin/Data/full_data/'
#data_root = '/home/user/server/home/d.grushevskaya1/projects/dron_maks/full_data/annotation/'
#data_root = '/home/dron_maks/full_data/annotation/'
# Определить тестовые и валидационные файлы вручную
TEST_FILES = [
	data_root + 'ch02_20200605121548-part 00000.json',
	]
	
VAL_FILES = [data_root + 'ch02_20200605114152-part 00000.json'] 


TRAIN_FILES = glob(data_root + '*.json')
#print(TEST_FILES, VAL_FILES, TRAIN_FILES)

for i in set(TEST_FILES + VAL_FILES):
	TRAIN_FILES.remove(i)


print(TRAIN_FILES, TEST_FILES, VAL_FILES)
