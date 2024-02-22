import numpy as np
import tkinter as tk
from tkinter import ttk


class Genetic:
    """
    Класс, реализующий генетический алгоритм
    """
    def __init__(self, num_generations, population_size, mutation_rate, encoding_type, low, high):
        """
        num_generations - количество поколений
        population_size - размер популяции одного поколения
        mutation_rate - шанс мутации в процентах
        encoding_type - вид кодирования. Здесь и далее рассматриваются два вида:
            1. Вещественное - 'real'
            2. Логарифмическое - 'logarithmic'
        low - нижняя граница поиска
        high - верхняя граница поиска
        """
        self.num_generations = num_generations
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.encoding_type = encoding_type
        self.low = low
        self.high = high
        self.population = self.initialize_population()

    def fitness_function(self, x, y):
        """
        Целевая функция, которую требуется минимизировать
        """
        return 2 * (x ** 3) + 4 * x * (y ** 3) - 10 * x * y + (y ** 2)

    def initialize_population(self):
        """
        Первоначальное случайное создание популяции в зависимости от кодировки
        """
        if self.encoding_type == 'real':
            return np.random.uniform(self.low, self.high, size=(self.population_size, 2))
        else:
            return np.random.uniform(0, 1, size=(self.population_size, 2))

    def enforce_bounds(self, individual):
        """
        Ограничения на значения особи: гены должны оставаться в пределах low и high
        """
        if self.encoding_type == 'real':
            return np.maximum(np.minimum(individual, self.high), self.low)
        else:
            return np.maximum(np.minimum(individual, 1), 0)
        
    def crossover(self, parent1, parent2):
        """
        Скрещивание особей - "обмен" генетической информацией между двумя родителями
        для создания потомства
        """
        crossover_point = np.random.randint(1, len(parent1))
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))

        return child1, child2

    def mutate(self, individual):
        """
        Операция мутации - некоторое случайное изменение в гены особи
        Помогает сохранять разнообразие в популяции и обеспечивает случайный поиск
        в пространстве решений
        """
        mutation_mask = np.random.rand(*individual.shape) < self.mutation_rate
        individual[mutation_mask] = np.random.randn(*individual.shape)[mutation_mask]
        return self.enforce_bounds(individual)


    def select_parents(self, population, fitness_values):
        """
        Метод турнирного выбора для определения особей, которые будут размножаться
        Лучшие особи выбираются в зависимости от значения целевой функции
        """
        tournament_size = 3
        indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = fitness_values[indices]
        parent_index = indices[np.argmin(tournament_fitness)]
        return population[parent_index]


def inserted(place, num):
    place.delete(0, tk.END)
    place.insert(0, str(num))
    
# Отображение данных в таблицу
def display_table(data):
    for row in tree.get_children():
        tree.delete(row)
    for row_data in data:
        tree.insert("", "end", values=row_data)


def calculate_genetic_algorithm(encoding_type):
  
    # Считывание данных с интерфейса
    mutation_rate = float(ProbMut.get()) / 100.0
    population_size = int(numChrom.get())
    num_generations = int(cntIt.get())
    low = float(minGen.get())
    high = float(maxGen.get())

    # Создание популяции
    genetic_algorithm = Genetic(num_generations, population_size, mutation_rate, encoding_type, low, high)
    generations_data = []
    
    # Главный процесс генетического алгоритма
    # Количество итераций = количество поколений
    for generation in range(num_generations):
        fitness_values = np.apply_along_axis(lambda x: genetic_algorithm.fitness_function(x[0], x[1]), 1, genetic_algorithm.population)

        # Сортировка в соответствии с наименьшими значениями целевой функции
        sorted_indices = np.argsort(fitness_values)
        genetic_algorithm.population = genetic_algorithm.population[sorted_indices]
        
        # Отбираются 50% особей
        selected_population = genetic_algorithm.population[:genetic_algorithm.population_size // 2]

        # Скрещивание и мутация отобранных особей
        new_population = []
        for i in range(genetic_algorithm.population_size // 2):
            parent1 = genetic_algorithm.select_parents(selected_population, fitness_values)
            parent2 = genetic_algorithm.select_parents(selected_population, fitness_values)
            child1, child2 = genetic_algorithm.crossover(parent1, parent2)
            child1 = genetic_algorithm.mutate(child1)
            child2 = genetic_algorithm.mutate(child2)
            new_population.extend([child1, child2])

        # Выделение лучшей особи
        genetic_algorithm.population = np.array(new_population)
        best_individual = genetic_algorithm.population[0]
        best_individual = genetic_algorithm.enforce_bounds(best_individual)
        best_fitness = genetic_algorithm.fitness_function(best_individual[0], best_individual[1])

        # Сохранение всех данных для визуализации хромосом данного поколения
        generations_data.append((generation + 1, best_fitness, best_individual[0], best_individual[1]))
    
    # Вывод результата
    if encoding_type == 'logarithmic':
        x_log = low * 10 ** (best_individual[0] * np.log10(1+ high/max(low, 1e-10)))
        y_log = low * 10 ** (best_individual[1] * np.log10(1+ high/max(low, 1e-10))) 
        coord_text = f"Координаты точки: ({-abs(x_log):.4f}, {y_log:.4f})"
        fitness_text = f"Функция: {-abs(genetic_algorithm.fitness_function(x_log, y_log)):.4f}"
    else:
        coord_text = f"Координаты точки: ({best_individual[0]:.4f}, {best_individual[1]:.4f})"
        fitness_text = f"Функция: {best_fitness:.4f}"
        
    canvas2.delete("all")
    canvas2.itemconfig(canvas2.create_text(197, 10, text=coord_text, font=("Arial", 10)), tags="result_text_coord")
    canvas2.itemconfig(canvas2.create_text(197, 40, text=fitness_text, font=("Arial", 10)), tags="result_text_fitness")
    display_table(generations_data)


root = tk.Tk()
root.title("Генетический алгоритм для поиска минимума функции")
root.geometry('800x450')


canvas = tk.Canvas(root, width=1000, height=1000, borderwidth=0, highlightthickness=0)
canvas.place(relx=0, rely=0.0)

line1 = canvas.create_line(0, 210, 350, 210)
line2 = canvas.create_line(350, 0, 350, 450)

frame_tree = ttk.Frame(root, borderwidth=2, relief="groove")
frame_tree.place(relx=0.465, rely=0.3)

columns = ("Номер", "Результат", "Ген 1", "Ген 2")
tree = ttk.Treeview(frame_tree, columns=columns, show="headings", height=10)

style = ttk.Style()
style.configure("Treeview.Heading", font=("Arial", 10))
style.configure("Treeview", font=("Arial", 9), rowheight=25)
style.layout("Treeview", [('Treeview.treearea', {'sticky': 'nswe'})])

for col in columns:
    tree.heading(col, text=col)
    tree.column(col, width=100)

tree.pack(expand=True, fill="both")

lbl = tk.Label(root, text="Предварительные настройки", font=("Arial", 11) )
lbl.place(relx=0.1, rely=0.01)

lblfunc = tk.Label(root, text="Функция:", font=("Arial", 10))
lblfunc.place(relx=0.027, rely=0.075)

lblfunc2 = tk.Label(root, text="2x[1]^3 + 4x[1]x[2]^3-10x[1]x[2]+x[2]^3", font=("Arial", 10))
lblfunc2.place(relx=0.12, rely=0.075)

lblk = tk.Label(root, text="Вероятность мутации, %:", font=("Arial", 10))
lblk.place(relx=0.027, rely=0.155)

ProbMut = tk.Entry(root,width=7)
ProbMut.insert(0, "20")
ProbMut.place(relx=0.35, rely=0.16)

lblk2 = tk.Label(root, text="Количество хромосом:", font=("Arial", 10))
lblk2.place(relx=0.027, rely=0.235)

numChrom = tk.Entry(root,width=7) 
numChrom.insert(0, "50")
numChrom.place(relx=0.35, rely=0.24)

lblk2 = tk.Label(root, text="Минимальное значение гена:", font=("Arial", 10))
lblk2.place(relx=0.027, rely=0.315)

minGen = tk.Entry(root,width=7) 
minGen.insert(0, "-50")
minGen.place(relx=0.35, rely=0.32)

lblk3 = tk.Label(root, text="Максимальное значение гена:", font=("Arial", 10))
lblk3.place(relx=0.027, rely=0.395)

maxGen = tk.Entry(root, width=7) 
maxGen.insert(0, "50")
maxGen.place(relx=0.35, rely=0.4)

lbl4 = tk.Label(root, text="Управление", font=("Arial", 11))
lbl4.place(relx=0.16, rely=0.48)

lbl5 = tk.Label(root, text="Количество поколений:", font=("Arial", 10))
lbl5.place(relx=0.027, rely=0.55)

cntIt = tk.Spinbox(root, from_=1, to=5000, width=5)
cntIt.place(relx=0.35, rely=0.555)

but1 = tk.Button(root, text="1", width=8, 
                 command=lambda: inserted(cntIt, 1), 
                 bg="#DDDDDD", activebackground="#CCCCCC", relief=tk.GROOVE)
but1.place(relx=0.03, rely=0.64)

but2 = tk.Button(root, text="10", width=8, 
                 command=lambda: inserted(cntIt, 10), 
                 bg="#DDDDDD", activebackground="#CCCCCC", relief=tk.GROOVE)
but2.place(relx=0.13, rely=0.64)

but3 = tk.Button(root, text="100", width=8, 
                 command=lambda: inserted(cntIt, 100),
                 bg="#DDDDDD", activebackground="#CCCCCC", relief=tk.GROOVE)
but3.place(relx=0.23, rely=0.64)

but4 = tk.Button(root, text="1000", width=8, 
                 command=lambda: inserted(cntIt, 1000),
                 bg="#DDDDDD", activebackground="#CCCCCC", relief=tk.GROOVE)
but4.place(relx=0.328, rely=0.64)

but5 = tk.Button(root, text="Рассчитать (логарифмическое кодирование)", width=42, 
                command=lambda: calculate_genetic_algorithm('logarithmic'),
                bg="#DDDDDD", activebackground="#CCCCCC", relief=tk.GROOVE)
but5.place(relx=0.03, rely=0.72)

but5 = tk.Button(root, text="Рассчитать (вещественное кодирование)", width=42, 
                command=lambda: calculate_genetic_algorithm('real'),
                bg="#DDDDDD", activebackground="#CCCCCC", relief=tk.GROOVE)
but5.place(relx=0.03, rely=0.8)

lbl6 = tk.Label(root, text="Результаты", font=("Arial", 11))
lbl6.place(relx=0.65, rely=0.01)

lbl5 = tk.Label(root, text="Лучшее решение достигается в точке:", font=("Arial", 10))
lbl5.place(relx=0.465, rely=0.075)

canvas2 = tk.Canvas(root, width=393, height=50, bg="white", borderwidth=1, highlightbackground="#CCCCCC", highlightthickness=2)
canvas2.place(relx=0.465, rely=0.14)

root.mainloop()
