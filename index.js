const tf = require("@tensorflow/tfjs")

const people = [
  { name: "Gabriel", age: 31, color: "blue", location: "London" },
  { name: "John", age: 25, color: "red", location: "New York" },
  { name: "Jane", age: 28, color: "green", location: "Los Angeles" },
]
const categories = {
  "Gabriel": "premium",
  "John": "standard",
  "Jane": "free",
}

// Categorização de cores
const colorCategories = ["blue", "red", "green"]
const locationCategories = ["London", "New York", "Los Angeles"]
const categoriesCategories = ["premium", "standard", "free"]
const peopleTensor = []
const categoriesTensor = []

// Normalização da idade
function normalizeAge(age, arrAges) {
  const max = Math.max(...arrAges)
  const min = Math.min(...arrAges)
  return (age - min) / (max - min)
}

async function trainModel(inputXs, outputYs) {
  const model = tf.sequential()

  // Primeira camada da rede:
  // Entrada de 7 posições (idae normalizada, 3 cores, 3 localizações)
  // 80 neurônios na primeira camada
  // A ReLU é uma função de ativação que retorna 0 se o valor for negativo e o valor se for positivo
  model.add(tf.layers.dense({ inputShape: [7], units: 80, activation: "relu" }))

  // Saida: 3 neuronios para as 3 categorias
  // A softmax é uma função de ativação que retorna a probabilidade de cada categoria
  model.add(tf.layers.dense({ units: 3, activation: "softmax" }))

  // optimizer adam (Adaptive Moment Estimation)
  // treinador moderno de redes neurais
  // ajusta os pesos de forma eficiente e adequada para o problema

  // loss: "categoricalCrossentropy" é a função de perda para classificação multi-classe
  // compara o resultado do modelo com o resultado esperado
  // quanto mais distante da previsão do modelo da resposta certa, maior o erro
  // Exemplo: classificação de imagem, recomendação, categorização de usuario

  // metrics: ["accuracy"] é a métrica de avaliação do modelo
  model.compile({ optimizer: "adam", loss: "categoricalCrossentropy", metrics: ["accuracy"] })

  // verbose: 0 para não mostrar o progresso do treinamento
  // epochs: quantidade de vezes que passará pelos dados da base de treinamento
  // shuffle: embaralha os dados da base de treinamento
  await model.fit(inputXs, outputYs, {
    verbose: 0,
    epochs: 100,
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(`Epoch ${epoch} - Loss: ${logs.loss} - Accuracy: ${logs.accuracy}`)
      }
    }
  })

  return model
}

async function predict(model, people) {
  const tfInput = tf.tensor2d(people)

  const pred = model.predict(tfInput)
  const predArray = await pred.array()
  return predArray[0].map((prob, index) => ({ prob, index }))
}

// Criação do tensor de pessoas
people.forEach(person => {
  const thisTensor = []
  thisTensor.push(normalizeAge(person.age, people.map(p => p.age)))

  colorCategories.forEach(color => {
    person.color === color ? thisTensor.push(1) : thisTensor.push(0)
  })

  locationCategories.forEach(location => {
    person.location === location ? thisTensor.push(1) : thisTensor.push(0)
  })

  peopleTensor.push(thisTensor)
})

for (const key in categories) {
  const thisCategory = categories[key]
  const thisTensor = []
  categoriesCategories.forEach(category => {
    thisCategory === category ? thisTensor.push(1) : thisTensor.push(0)
  })
  categoriesTensor.push(thisTensor)
}

//console.log(peopleTensor)
//console.log(categoriesTensor)

const inputXs = tf.tensor2d(peopleTensor)
const outputYs = tf.tensor2d(categoriesTensor);

(async () => {
  const model = await trainModel(inputXs, outputYs)

  const people = {
    name: "zé",
    age: 28,
    color: "green",
    location: "London"
  }

  const tensorPeople = [
    [
      0.2, // age
      0, // color blue
      1, // color red
      0, // color green
      0, // location London
      0, // location New York
      1, // location Los Angeles
    ]
  ]

  const predictions = await predict(model, tensorPeople)
  const results = predictions
    .sort((a, b) => b.prob - a.prob)
    .map(p => `${categoriesCategories[p.index]} - ${(p.prob * 100).toFixed(2)}%`)

  console.log(results)
})();

//inputXs.print()
//inputYs.print()


