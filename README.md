# Regression-lineaire-numpy
Nous allons développer un algorithme de descente de gradient pour résoudre un problème de régression linéaire avec Python et sa librairie Numpy.
## 1. Importer les packages Numpy et Matplotlib.pyplot
Avant toute chose, il est nécessaire d’importer les packages Numpy et Matplotlib.pyplot. Numpy permet de créer des matrices et effectuer des opérations mathématiques. Matplotlib permet de créer des graphiques pour observer facilement notre dataset ainsi que le modèle construit à partir de celui-ci.
<p align="center">
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/31/NumPy_logo_2020.svg/1024px-NumPy_logo_2020.svg.png" width="400px"/>
</p>

<p align="center">
<img src="https://matplotlib.org/stable/_images/sphx_glr_logos2_003_2_0x.png" width="400px"/>
</p>
## 2. Génération d’un dataset (x, y) linéaire
Avec la fonction linspace de Numpy, nous créons un tableau de données (x, y) qui présente une tendance linéaire. La fonction random.randn permet d’ajouter un “bruit” aléatoire normal aux données. Pour effectuer un calcul matriciel correct, il est important de confier 2 dimensions (100 lignes, 1 colonne) à ces tableaux en utilisant la fonction reshape(100, 1).
Une fois le dataset généré, il faut ajouter une colonne de biais au tableau X, c’est-à-dire un colonne de 1, pour le développement du futur modele linéaire f(x) = a\timesx + b\times1, puis initialiser des parametres a,b dans un vecteur theta.
## 3. Développement des fonctions de Descente de gradient
Pour développer un modèle linéaire (ou polynomial !) avec la déscente de gradient, il faut implémenter les 4 fonctions clefs suivantes :

la fonction de notre modèle : f(x) = X.\theta
la fonction Cout : J(\theta) = \frac{1}{2m} \sum (X.\theta - Y)^2
le gradient : \frac{\partial J(\theta)}{\partial \theta} = \frac{1}{m} X^T.(X.\theta - Y)
la descente de gradient : \theta = \theta - \alpha \times \frac{\partial J(\theta)}{\partial \theta}
## 4. Entrainement du modèle
Une fois les fonctions ci-dessus implémentées, il suffit d’utiliser la fonction gradient_descent en indiquant un nombre d’itérations ainsi qu’un learning rate, et la fonction retournera les paramètres du modèle après entrainement, sous forme de la variable theta_final.

Vous pouvez ensuite visualiser votre modèle grâce à Matplotlib.
Pour finir, vous pouvez visualiser l’évolution de la descente de gradient en créant un graphique qui trace la fonction_cout en fonction du nombre d’itération. Si votre descente de gradient a bien fonctionné, vous devez obtenir une courbe qui diminue progressivement jusqu’à converger vers un certain minimum. Si vous n’observez pas de stabilisation, alors cela signifie que le modèle n’a pas terminé son apprentissage et qu’il faut soit augmenter le nombre d’itérations de la descente de gradient ou bien le pas (learning_rate).

