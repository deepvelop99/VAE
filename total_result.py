import matplotlib.pyplot as plt
import torch, torchvision
import numpy as np

def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('./datasets',
                transform=torchvision.transforms.ToTensor(),
                download=True),
            batch_size=128,
            shuffle=True)
    vae = torch.load("result/vae_model.pt", map_location=device)
    vae.eval()

    def plot_latent(autoencoder, data, num_batches=100):
        for i, (x, y) in enumerate(data):
            z = autoencoder.encoder(x.to(device))
            z = z.to('cpu').detach().numpy()
            plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
            if i > num_batches:
                plt.colorbar()
                break
        plt.show()

    plot_latent(vae, data)

    def plot_reconstructed(autoencoder, r0=(-5, 10), r1=(-10, 5), n=12):
        w = 28
        img = np.zeros((n*w, n*w))
        for i, y in enumerate(np.linspace(*r1, n)):
            for j, x in enumerate(np.linspace(*r0, n)):
                z = torch.Tensor([[x, y]]).to(device)
                x_hat = autoencoder.decoder(z)
                x_hat = x_hat.reshape(28, 28).to('cpu').detach().numpy()
                img[(n-1-i)*w:(n-1-i+1)*w, j*w:(j+1)*w] = x_hat
        plt.imshow(img, extent=[*r0, *r1])
        plt.show()

    plot_reconstructed(vae)

    def interpolate(autoencoder, x_1, x_2, n=12):
        z_1 = autoencoder.encoder(x_1)
        z_2 = autoencoder.encoder(x_2)
        z = torch.stack([z_1 + (z_2 - z_1)*t for t in np.linspace(0, 1, n)])
        interpolate_list = autoencoder.decoder(z)
        interpolate_list = interpolate_list.to('cpu').detach().numpy()

        w = 28
        img = np.zeros((w, n*w))
        for i, x_hat in enumerate(interpolate_list):
            img[:, i*w:(i+1)*w] = x_hat.reshape(28, 28)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.show()

    x, y = data.__iter__().__next__() # hack to grab a batch
    x_1 = x[y == 1][1].to(device) # find a 1
    x_2 = x[y == 0][1].to(device) # find a 0

    interpolate(vae, x_1, x_2, n=20)
    interpolate(vae, x_1, x_2, n=20)

    from PIL import Image

    def interpolate_gif(autoencoder, filename, x_1, x_2, n=100):
        z_1 = autoencoder.encoder(x_1)
        z_2 = autoencoder.encoder(x_2)

        z = torch.stack([z_1 + (z_2 - z_1)*t for t in np.linspace(0, 1, n)])

        interpolate_list = autoencoder.decoder(z)
        interpolate_list = interpolate_list.to('cpu').detach().numpy()*255

        images_list = [Image.fromarray(img.reshape(28, 28)).resize((256, 256)) for img in interpolate_list]
        images_list = images_list + images_list[::-1] # loop back beginning

        images_list[0].save(
            f'{filename}.gif',
            save_all=True,
            append_images=images_list[1:],
            loop=1)
        
    interpolate_gif(vae, "vae", x_1, x_2)

test()