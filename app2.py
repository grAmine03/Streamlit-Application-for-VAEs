import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

# Configuration de la page Streamlit
st.set_page_config(page_title="Explorateur de VAE", page_icon="🧠", layout="wide")

# Titre et introduction
st.title("🧠 Explorateur de Variational Autoencoders (VAE)")
st.markdown(
    """
Cette application vous permet d'explorer les Variational Autoencoders (VAE) en modifiant leurs hyperparamètres
et en observant comment ils affectent l'apprentissage et la génération d'images.
"""
)


# Définition du modèle VAE
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()

        # Encodeur
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)

        # Décodeur
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        mu = self.fc_mu(h1)
        log_var = self.fc_var(h1)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, log_var = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var


# Fonction pour calculer la perte
def loss_function(recon_x, x, mu, log_var, beta=1.0):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction="sum")
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + beta * KLD


# Fonction d'entraînement
def train(model, train_loader, optimizer, device, beta):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, log_var = model(data)
        loss = loss_function(recon_batch, data, mu, log_var, beta)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    return train_loss / len(train_loader.dataset)


# Fonction de test
def test(model, test_loader, device, beta):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, log_var = model(data)
            test_loss += loss_function(recon_batch, data, mu, log_var, beta).item()

    test_loss /= len(test_loader.dataset)
    return test_loss


# Visualisation des images originales et reconstruites
def visualize_reconstruction(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        # Obtenir un batch de données
        data, _ = next(iter(test_loader))
        data = data.to(device)

        # Reconstruire les images
        recon_batch, _, _ = model(data)

        # Préparer l'affichage
        comparison = torch.cat(
            [data[:8].view(-1, 1, 28, 28), recon_batch[:8].view(-1, 1, 28, 28)]
        )

        # Convertir pour matplotlib
        comparison = comparison.cpu().numpy()

        fig, axes = plt.subplots(2, 8, figsize=(12, 3))
        for i in range(8):
            # Images originales
            axes[0, i].imshow(comparison[i][0], cmap="gray")
            axes[0, i].axis("off")

            # Images reconstruites
            axes[1, i].imshow(comparison[i + 8][0], cmap="gray")
            axes[1, i].axis("off")

        fig.tight_layout()
        return fig


# Visualisation de l'espace latent
def visualize_latent_space(model, test_loader, device, latent_dim):
    model.eval()
    with torch.no_grad():
        # Obtenir des données
        data_list, label_list = [], []
        for data, labels in test_loader:
            data_list.append(data)
            label_list.append(labels)
            if len(data_list) >= 10:  # Limiter à 10 batches pour la vitesse
                break

        data = torch.cat(data_list, dim=0).to(device)
        labels = torch.cat(label_list, dim=0).to(device)

        # Encoder les données
        mu, _ = model.encode(data.view(-1, 784))
        z = mu.cpu().numpy()
        labels = labels.cpu().numpy()

        # Si la dimension latente > 2, utiliser t-SNE ou PCA
        if latent_dim > 2:
            st.warning(
                "Dimension latente > 2. Affichage des 2 premières dimensions uniquement."
            )
            z = z[:, :2]  # Utilisez seulement les 2 premières dimensions

        # Tracer
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(z[:, 0], z[:, 1], c=labels, cmap="tab10", alpha=0.6, s=5)

        legend = ax.legend(*scatter.legend_elements(), title="Chiffres")
        ax.add_artist(legend)

        ax.set_title("Visualisation de l'espace latent")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")

        fig.tight_layout()
        return fig


# Génération d'images à partir de l'espace latent
def generate_from_latent(model, device, latent_dim, n_samples=10):
    model.eval()
    with torch.no_grad():
        # Échantillonner aléatoirement de l'espace latent
        z = torch.randn(n_samples, latent_dim).to(device)

        # Décoder
        sample = model.decode(z).cpu().numpy()

        # Afficher les images générées
        fig, axes = plt.subplots(1, n_samples, figsize=(12, 1.5))
        for i in range(n_samples):
            axes[i].imshow(sample[i].reshape(28, 28), cmap="gray")
            axes[i].axis("off")

        fig.tight_layout()
        return fig


# Interface utilisateur Streamlit
def main():
    # Barre latérale pour les hyperparamètres
    st.sidebar.header("Hyperparamètres")

    # Paramètres du VAE
    latent_dim = st.sidebar.slider("Dimension de l'espace latent", 2, 100, 20)
    hidden_dim = st.sidebar.slider("Dimension de la couche cachée", 200, 800, 400)
    beta = st.sidebar.slider("Beta (poids du terme KL)", 0.1, 5.0, 1.0, 0.1)

    # Paramètres d'entraînement
    learning_rate = st.sidebar.slider(
        "Taux d'apprentissage", 0.0001, 0.01, 0.001, format="%.4f"
    )
    batch_size = st.sidebar.selectbox("Taille du batch", [32, 64, 128, 256], index=1)
    epochs = st.sidebar.slider("Nombre d'époques", 1, 20, 5)

    # Informations sur le dataset
    st.sidebar.header("Dataset")
    st.sidebar.info(
        """
    **MNIST** : 60 000 images d'entraînement et 10 000 images de test
    de chiffres manuscrits (0-9) en noir et blanc de dimensions 28x28 pixels.
    """
    )

    # À propos des VAE
    with st.expander("En savoir plus sur les VAE"):
        st.markdown(
            """
        ## Qu'est-ce qu'un VAE ?

        Un Variational Autoencoder (VAE) est un type de modèle génératif qui apprend à représenter
        des données dans un espace latent continu à partir duquel de nouvelles données peuvent être générées.

        ### Composants principaux :

        1. **Encodeur** : Transforme les données d'entrée en une distribution dans l'espace latent
        2. **Espace latent** : Représentation compressée des données (moyenne μ et variance σ²)
        3. **Échantillonnage** : Utilise la "reparametrization trick" pour permettre la rétropropagation
        4. **Décodeur** : Reconstruit les données originales à partir de l'espace latent

        ### Fonction de perte :

        La fonction de perte d'un VAE comporte deux termes :
        - **Erreur de reconstruction** : Mesure la différence entre l'entrée et la sortie reconstruite
        - **Divergence KL** : Force la distribution latente à se rapprocher d'une distribution normale

        Le paramètre β contrôle l'importance relative de ces deux termes.
        """
        )

    # Actions principales
    col1, col2 = st.columns(2)
    with col1:
        start_training = st.button("Entraîner le modèle")
    with col2:
        show_examples = st.button("Voir des exemples sans entraînement")

    # Chargement des données
    @st.cache_resource
    def load_data(batch_size):
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.MNIST(
            "./data", train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST("./data", train=False, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        return train_loader, test_loader

    # Préchargement des données
    with st.spinner("Chargement des données..."):
        train_loader, test_loader = load_data(batch_size)

    # Détection du device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_info = f"Utilisation de {'GPU' if torch.cuda.is_available() else 'CPU'}"
    st.sidebar.info(device_info)

    # Modèle préentraîné (pour démo rapide)
    @st.cache_resource
    def get_pretrained_model():
        model = VAE(latent_dim=20).to(device)
        try:
            model.load_state_dict(
                torch.load("vae_model_pretrained.pth", map_location=device)
            )
            return model
        except BaseException:
            # Si aucun modèle préentraîné n'est disponible, on continue
            return None

    pretrained_model = get_pretrained_model()

    # Afficher des exemples avec le modèle préentraîné
    if show_examples and pretrained_model is not None:
        st.subheader("Exemples avec un modèle préentraîné")

        # Afficher les reconstructions
        st.markdown("### Reconstructions d'images")
        fig_recon = visualize_reconstruction(pretrained_model, test_loader, device)
        st.pyplot(fig_recon)

        # Afficher l'espace latent
        st.markdown("### Visualisation de l'espace latent")
        fig_latent = visualize_latent_space(pretrained_model, test_loader, device, 20)
        st.pyplot(fig_latent)

        # Générer des images
        st.markdown("### Images générées depuis l'espace latent")
        fig_gen = generate_from_latent(pretrained_model, device, 20)
        st.pyplot(fig_gen)

    # Entraînement du modèle
    if start_training:
        st.header("Entraînement du VAE")

        # Initialisation du modèle
        model = VAE(input_dim=784, hidden_dim=hidden_dim, latent_dim=latent_dim).to(
            device
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Préparation pour suivre les pertes
        train_losses = []
        test_losses = []

        # Affichage de la progression
        progress_bar = st.progress(0)
        status_text = st.empty()
        loss_chart = st.line_chart()

        # Entraînement par époque
        for epoch in range(1, epochs + 1):
            start_time = time.time()

            # Entraînement
            train_loss = train(model, train_loader, optimizer, device, beta)
            train_losses.append(train_loss)

            # Validation
            test_loss = test(model, test_loader, device, beta)
            test_losses.append(test_loss)

            # Mise à jour de l'interface
            epoch_time = time.time() - start_time
            status_text.text(
                f"Époque {epoch}/{epochs} - Perte d'entraînement: {train_loss:.4f}, "
                f"Perte de test: {test_loss:.4f}, Temps: {epoch_time:.2f}s"
            )

            progress_bar.progress(epoch / epochs)

            # Mise à jour du graphique de perte
            loss_chart.add_rows(
                {"Perte d'entraînement": train_loss, "Perte de test": test_loss}
            )

            # Affichage des reconstructions après certaines époques
            if epoch % max(1, epochs // 5) == 0 or epoch == epochs:
                st.subheader(f"Résultats après l'époque {epoch}")

                # Afficher les reconstructions
                st.markdown("#### Reconstructions d'images")
                fig_recon = visualize_reconstruction(model, test_loader, device)
                st.pyplot(fig_recon)

                # Afficher l'espace latent
                st.markdown("#### Visualisation de l'espace latent")
                fig_latent = visualize_latent_space(
                    model, test_loader, device, latent_dim
                )
                st.pyplot(fig_latent)

                # Générer des images
                st.markdown("#### Images générées depuis l'espace latent")
                fig_gen = generate_from_latent(model, device, latent_dim)
                st.pyplot(fig_gen)

        # Sauvegarde du modèle
        torch.save(model.state_dict(), "vae_model_trained.pth")
        st.success("Entraînement terminé ! Le modèle a été sauvegardé.")

        # Bouton pour explorer l'espace latent interactif
        if st.button("Explorer l'espace latent interactivement"):
            st.subheader("Explorateur d'espace latent")
            st.markdown(
                """
            Déplacez les curseurs pour générer des images à partir de différents points de l'espace latent.
            Chaque dimension correspond à une caractéristique apprise par le modèle.
            """
            )

            # Créer des curseurs pour chaque dimension (limité à 10 pour la lisibilité)
            max_sliders = min(10, latent_dim)
            z = torch.zeros(1, latent_dim).to(device)

            for i in range(max_sliders):
                z[0, i] = st.slider(f"Dimension {i + 1}", -3.0, 3.0, 0.0, 0.1)

            # Générer et afficher l'image
            with torch.no_grad():
                image = model.decode(z).cpu().numpy()

                fig, ax = plt.subplots(figsize=(3, 3))
                ax.imshow(image[0].reshape(28, 28), cmap="gray")
                ax.axis("off")
                st.pyplot(fig)

            if latent_dim > max_sliders:
                st.info(
                    f"Seules les {max_sliders} premières dimensions sont affichées pour la lisibilité."
                )

    # Interpolation dans l'espace latent (disponible si un modèle existe)
    if (pretrained_model is not None) or ("model" in locals()):
        with st.expander("Interpolation dans l'espace latent"):
            st.markdown(
                """
            Cette section vous permet de visualiser l'interpolation entre deux points dans l'espace latent.
            Cela montre comment le modèle génère des transitions fluides entre les chiffres.
            """
            )

            # Utiliser le modèle disponible
            model_to_use = model if "model" in locals() else pretrained_model

            # Nombre de points d'interpolation
            steps = st.slider("Nombre d'étapes d'interpolation", 5, 20, 10)

            if st.button("Générer interpolation"):
                # Générer deux points aléatoires dans l'espace latent
                z1 = torch.randn(1, model_to_use.fc_mu.out_features).to(device)
                z2 = torch.randn(1, model_to_use.fc_mu.out_features).to(device)

                # Créer les points d'interpolation
                alphas = np.linspace(0, 1, steps)
                z_interp = torch.stack(
                    [z1 * (1 - alpha) + z2 * alpha for alpha in alphas]
                )

                # Décoder les points
                with torch.no_grad():
                    interp_images = (
                        model_to_use.decode(z_interp.squeeze()).cpu().numpy()
                    )

                # Afficher les images
                fig, axes = plt.subplots(1, steps, figsize=(12, 2))
                for i in range(steps):
                    axes[i].imshow(interp_images[i].reshape(28, 28), cmap="gray")
                    axes[i].axis("off")

                st.pyplot(fig)

    # Section pour comparer différents modèles
    with st.expander("Comparaison de différents VAE"):
        st.markdown(
            """
        Cette section vous permet de comparer différents modèles VAE entraînés avec des hyperparamètres variables.
        Cliquez sur le bouton ci-dessous pour entraîner rapidement plusieurs modèles avec différentes dimensions latentes.
        """
        )

        if st.button("Entraîner et comparer différents modèles (rapide)"):
            latent_dims = [2, 5, 10, 20]
            beta_values = [0.5, 1.0, 2.0]

            # Créer une grille pour les résultats
            grid_cols = st.columns(len(latent_dims))

            for i, lat_dim in enumerate(latent_dims):
                with grid_cols[i]:
                    st.markdown(f"#### Dim. latente = {lat_dim}")

                    # Créer et entraîner un modèle rapidement (1 époque)
                    quick_model = VAE(latent_dim=lat_dim).to(device)
                    optimizer = torch.optim.Adam(quick_model.parameters(), lr=0.001)

                    with st.spinner(f"Entraînement rapide (dim={lat_dim})..."):
                        # Entraînement rapide (1 époque)
                        train(quick_model, train_loader, optimizer, device, beta=1.0)

                    # Afficher les reconstructions
                    fig_recon = visualize_reconstruction(
                        quick_model, test_loader, device
                    )
                    st.pyplot(fig_recon)

                    # Afficher l'espace latent
                    fig_latent = visualize_latent_space(
                        quick_model, test_loader, device, lat_dim
                    )
                    st.pyplot(fig_latent)

            # Comparer les différentes valeurs de beta
            st.markdown("### Impact du paramètre β")
            beta_cols = st.columns(len(beta_values))

            for i, beta_val in enumerate(beta_values):
                with beta_cols[i]:
                    st.markdown(f"#### β = {beta_val}")

                    # Créer et entraîner un modèle rapidement (1 époque)
                    quick_model = VAE(latent_dim=10).to(device)
                    optimizer = torch.optim.Adam(quick_model.parameters(), lr=0.001)

                    with st.spinner(f"Entraînement rapide (β={beta_val})..."):
                        # Entraînement rapide (1 époque)
                        train(
                            quick_model, train_loader, optimizer, device, beta=beta_val
                        )

                    # Afficher les reconstructions
                    fig_recon = visualize_reconstruction(
                        quick_model, test_loader, device
                    )
                    st.pyplot(fig_recon)

                    # Générer des images
                    fig_gen = generate_from_latent(quick_model, device, 10, n_samples=5)
                    st.pyplot(fig_gen)

    # Ressources supplémentaires
    st.header("Ressources supplémentaires")
    st.markdown(
        """
    ### Articles et tutoriels sur les VAE :
    - [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) (Article original)
    - [Tutorial on Variational Autoencoders](https://arxiv.org/abs/1606.05908)
    - [Understanding Variational Autoencoders](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73)

    ### Applications courantes des VAE :
    - Génération d'images
    - Compression de données
    - Détection d'anomalies
    - Génération de molécules
    - Synthèse de visages
    """
    )


if __name__ == "__main__":
    main()
