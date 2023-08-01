import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time

class TorchAdapter(object):

    @classmethod
    def split_data_train_validation_test(cls, full_dataset, validation_ratio, test_ratio, shuffle_data, params):
        dataset_size = len(full_dataset)
        indices = list(range(dataset_size))

        if shuffle_data:
            random_seed = int(time.time())
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        val_split = int(np.floor((1 - (validation_ratio + test_ratio)) * dataset_size))
        test_split = int(np.floor((1 - test_ratio) * dataset_size))
        train_indices, val_indices, test_indices = indices[:val_split], indices[val_split:test_split], indices[test_split:]

        ## Define training and validation data sampler
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        ## Define training and validation data loader
        train_loader = DataLoader(full_dataset, **params, sampler=train_sampler)
        val_loader = DataLoader(full_dataset, **params, sampler=valid_sampler)
        test_loader = DataLoader(full_dataset, **params, sampler=test_sampler)

        return train_loader, val_loader, test_loader


    @classmethod
    def get_training_device(cls):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        return device


    @classmethod
    def get_item_from_dataloader(cls, dataloader, batch, index):
        batch_index = index // batch
        img_index = index % batch
        current_index = 0
        for data in dataloader:
            if current_index == batch_index:
                return data, img_index
            current_index += 1
        return None, None

    @classmethod
    def visualize_all_bands(cls, image, no_bands, axis_rows, axis_cols):
        f, (ax) = plt.subplots(axis_rows, axis_cols, figsize=(15, 5))

        band_index = 0
        if axis_rows == 1 or axis_cols == 1:
            rows_cols = axis_rows * axis_cols
            for i in range(rows_cols):
                if band_index >= no_bands:
                    ax[i].axis('off')
                    continue
                ax[i].set_title("Band" + str((band_index + 1)))
                ax[i].imshow(image[band_index])
                band_index += 1

        else:
            for i in range(axis_rows):
                for j in range(axis_cols):
                    if band_index >= no_bands:
                        ax[i][j].axis('off')
                        continue
                    ax[i][j].set_title("Band" + str((band_index + 1)))
                    ax[i][j].imshow(image[band_index])
                    band_index += 1

    @classmethod
    def visualize_single_band_image(cls, image, title):
        f, (ax) = plt.subplots()
        ax.set_title(title)
        ax.imshow(image)


    @classmethod
    def show_bar_chart(cls, class_ids, probabilities):
        categories = []
        values = []
        for i in range(len(class_ids)):
            categories.append(class_ids[i])
            values.append(probabilities[i] * 100)

        plt.figure(figsize=(16, 6))
        bars = plt.bar(categories, values)

        # set title with increased font size
        plt.title('Classification Probability', fontsize=18)

        # set x and y axis labels with increased font size
        plt.xlabel('Categories', fontsize=14)
        plt.ylabel('% of Probabilities', fontsize=14)

        # increase font size of tick labels
        plt.xticks(fontsize=9)
        plt.yticks(fontsize=9)

        plt.show()

    @classmethod
    def show_pie_chart(cls, class_ids, probabilities):
        categories = []
        values = []
        explode = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        for i in range(len(class_ids)):
            categories.append(class_ids[i])
            values.append(probabilities[i] * 100)

        fig1, ax1 = plt.subplots(figsize=(10, 6))
        wedges, texts = ax1.pie(values, explode=explode,
                shadow=False, startangle=90)
        ax1.axis('equal')

        total = sum(values)
        percentages = [(100 * size / total) for size in values]
        labels = ['{} ({:.1f}%)'.format(label, percentage) for label, percentage in zip(categories, percentages)]

        plt.legend(wedges, labels, title="Classes", loc="best")  # Add a legend
        plt.show()


    @classmethod
    def visualize_bands_and_probabilities(cls, image, class_ids, probabilities):
        fig = plt.figure(figsize=(15, 5))  # Adjust the size as necessary
        gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 2])

        # Top left plot
        ax0 = plt.subplot(gs[0])
        ax0.set_title("Band" + str((1)))
        ax0.imshow(image[0])

        # Bottom left plot
        ax1 = plt.subplot(gs[1])
        ax1.set_title("Band" + str((2)))
        ax1.imshow(image[1])

        # Right plot (single row, larger)
        ax2 = plt.subplot(gs[2])
        ax2.set_title("Band" + str((3)))
        ax2.imshow(image[2])

        #ax3 = plt.subplot(gs[3])
        #ax3.axis('off')

        ax4 = plt.subplot(gs[3])
        categories = []
        values = []
        explode = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        for j in range(len(class_ids)):
            categories.append(class_ids[j])
            values.append(probabilities[j] * 100)

        wedges, texts = ax4.pie(values, explode=explode, shadow=False, startangle=90)
        ax4.axis('equal')

        total = sum(values)
        percentages = [(100 * size / total) for size in values]
        labels = ['{} ({:.1f}%)'.format(label, percentage) for label, percentage in
                  zip(categories, percentages)]

        ax4.legend(wedges, labels, title="Classes", loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)  # Add a legend

        # Display the figure
        #plt.tight_layout()
        plt.show()


