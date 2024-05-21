import cv2
import numpy as np
import random
import os
import matplotlib.pyplot as plt

class Individual:
    def __init__(self, source_image, num_genes, mutation_prob, mutation_type):
        self.num_genes = num_genes
        self.mutation_prob = mutation_prob
        self.mutation_type = mutation_type
        self.fitness = float('-inf')
        radius_value_init = random.randint(1, 45)
        self.genes = [{
                'x': random.randint(-radius_value_init+1, 180+radius_value_init-1),
                'y': random.randint(-radius_value_init+1, 180+radius_value_init-1),
                'radius': radius_value_init,
                'R': random.randint(0, 255),
                'G': random.randint(0, 255),
                'B': random.randint(0, 255),
                'A': random.uniform(0, 1)
            } for _ in range(self.num_genes)]
        self.genes.sort(key=lambda x: -x['radius'])
        self.evaluate_individual(source_image)

    def random_gene(self, gene_index):
        if self.mutation_type == 'unguided':
            radius_value = random.randint(1, 45)
            return {
                'x': random.randint(-radius_value+1, 180+radius_value-1),
                'y': random.randint(-radius_value+1, 180+radius_value-1),
                'radius': radius_value,
                'R': random.randint(0, 255),
                'G': random.randint(0, 255),
                'B': random.randint(0, 255),
                'A': random.uniform(0, 1)
            }
        elif self.mutation_type == 'guided':
            radius_value = max(0, min(45, random.randint(self.genes[gene_index]['radius'] -10, self.genes[gene_index]['radius'] +10)))
            return {
                'x': max(-radius_value+1, min(180+radius_value-1, random.randint(self.genes[gene_index]['x'] -45, self.genes[gene_index]['x']+45))),
                'y': max(-radius_value+1, min(180+radius_value-1, random.randint(self.genes[gene_index]['y'] -45, self.genes[gene_index]['y']+45))),
                'radius': radius_value,
                'R': min(255, max(0,random.randint(self.genes[gene_index]['R']-64,self.genes[gene_index]['R']+64))),
                'G': min(255, max(0,random.randint(self.genes[gene_index]['G']-64,self.genes[gene_index]['G']+64))),
                'B': min(255, max(0,random.randint(self.genes[gene_index]['B']-64,self.genes[gene_index]['B']+64))),
                'A': min(1, max(0, random.uniform(self.genes[gene_index]['A']-0.25, self.genes[gene_index]['A']+0.25)))
            }

    def evaluate_individual(self, source_image):
        image = np.ones((180, 180, 3), np.int32) * 255
        for gene in self.genes:
            overlay = image.copy()
            color = (gene['B'], gene['G'], gene['R'])
            cv2.circle(overlay, (gene['x'], gene['y']), gene['radius'], color, -1)
            alpha = gene['A']
            image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        self.fitness = -np.sum(np.square(source_image - image))

    def save_image(self, file_name, folder_name):
      if not os.path.exists(folder_name):
          os.makedirs(folder_name)
      image = np.ones((180, 180, 3), np.uint8) * 255
      for gene in self.genes:
          overlay = image.copy()
          color = (gene['B'], gene['G'], gene['R'])  # OpenCV uses BGR color format
          cv2.circle(overlay, (gene['x'], gene['y']), gene['radius'], color, -1)
          alpha = gene['A']
          image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
      # Save image using OpenCV
      cv2.imwrite(os.path.join(folder_name, file_name), image)

class Population:
    def __init__(self, num_inds, num_genes, source_image, tm_size, frac_elites, frac_parents, mutation_prob, mutation_type):
        self.num_inds = num_inds
        self.num_genes = num_genes
        self.source_image = source_image
        self.tm_size = tm_size
        self.frac_elites = frac_elites
        self.frac_parents = frac_parents
        self.mutation_prob = mutation_prob
        self.mutation_type = mutation_type
        self.num_elites = int(num_inds * frac_elites)
        self.inds = [Individual(source_image, num_genes, mutation_prob, mutation_type) for _ in range(num_inds)]
        self.inds.sort(key=lambda x: x.fitness, reverse=True)
        self.elites = self.inds[:self.num_elites]
        self.fitness_history = []
        self.num_parents = int(frac_parents*num_inds)
    def run_generation(self):
        # Tournament selection for parents
        potential_parents = self.inds[self.num_elites:]
        parents = []
        while len(parents) < self.num_parents:
            tournament = random.sample(potential_parents, self.tm_size)
            tournament.sort(key=lambda x: x.fitness, reverse=True)
            parents.append(tournament[0])
        # Crossover and generate children
        children = []
        while len(children) < self.num_inds - self.num_elites:
            if len(parents) > 1:
                random.shuffle(parents)
                parent1, parent2 = parents[0], parents[1]
                child1, child2 = Individual(self.source_image, self.num_genes, self.mutation_prob, self.mutation_type), Individual(self.source_image, self.num_genes, self.mutation_prob, self.mutation_type)
                for i in range(self.num_genes):
                    if random.random() < 0.5:
                        child1.genes[i], child2.genes[i] = parent1.genes[i], parent2.genes[i]
                    else:
                        child1.genes[i], child2.genes[i] = parent2.genes[i], parent1.genes[i]
                children.append(child1)
                children.append(child2)

        # Mutation
        for child in children:
            if random.random() < self.mutation_prob:
                gene_index = random.randint(0, self.num_genes - 1)
                child.genes[gene_index] = child.random_gene(gene_index)

        # Update population
        self.inds = self.elites+ children
        
        for ind in self.inds:
            ind.genes.sort(key=lambda x: -x['radius'])
            ind.evaluate_individual(self.source_image)
        self.inds.sort(key=lambda x: x.fitness, reverse=True)
        self.elites = self.inds[:self.num_elites]
        self.fitness_history.append(self.elites[0].fitness)

def plot_fitness_history(fitness_history, filename, title="Fitness Over Generations"):
    plt.figure(figsize=(10, 5))
    plt.plot(fitness_history, label='Best Fitness')
    plt.title(title)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def main(num_inds, num_genes, tm_size, frac_elites, frac_parents, mutation_prob, mutation_type):
    source_image = cv2.imread("painting.png", cv2.IMREAD_COLOR)
    source_image = cv2.resize(source_image, (180, 180))
    

    pop = Population(num_inds=num_inds, num_genes=num_genes, source_image=source_image, tm_size = tm_size, frac_elites=frac_elites, frac_parents=frac_parents, mutation_prob=mutation_prob, mutation_type=mutation_type)
    num_generations = 10000
    for generation in range(num_generations):
        pop.run_generation()
        
        if (generation + 1) % 1000 == 1 or generation+1 == 10000:
            # Save the image of the best elite individual
            pop.elites[0].save_image(f"elite_gen_{generation + 1}.png",folder_name=f"results: {num_inds}, {num_genes}, {tm_size}, {frac_elites}, {frac_parents},{mutation_prob},{mutation_type}")
            # Print the fitness of the best individual in the population
            print(f"Generation {generation + 1}: Best Fitness = {pop.elites[0].fitness}")

    plot_fitness_history(pop.fitness_history, title = "Fitness from Generation 1 to 10000", filename=f"result_plots: {num_inds}, {num_genes}, {tm_size}, {frac_elites}, {frac_parents},{mutation_prob},{mutation_type}.png")
    plot_fitness_history(pop.fitness_history[999:], title = "fitness_1000_to_10000.png",filename=f"result_plots 1000_to_10000: {num_inds}, {num_genes}, {tm_size}, {frac_elites}, {frac_parents},{mutation_prob},{mutation_type}.png")
for j in range(3):
    num_inds_list = [5, 10, 40]
    if num_inds_list[j] == 5:
        if __name__ == "__main__":
            main(num_inds=num_inds_list[j], num_genes=50, tm_size=2, frac_elites=0.2, frac_parents=0.6, mutation_prob=0.2, mutation_type='guided')
    else:
        if __name__ == "__main__":
            main(num_inds=num_inds_list[j], num_genes=50, tm_size=5, frac_elites=0.2, frac_parents=0.6, mutation_prob=0.2, mutation_type='guided')

    