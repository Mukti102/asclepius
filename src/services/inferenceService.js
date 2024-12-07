const tf = require('@tensorflow/tfjs-node');
const InputError = require('../exceptions/inputError');
 
async function predictClassification(model, image) {
  try{

    const tensor = tf.node
      .decodeJpeg(image)
      .resizeNearestNeighbor([224, 224])
      .expandDims()
      .toFloat()
   
    const prediction = await  model.predict(tensor).data();
    const result = prediction[0] > 0.5 ? "Cancer" : 'Non-Cancer';
    
    let label,suggestion

    if(result === 'Cancer'){
        label = "Cancer"
        suggestion = "Segera periksa ke dokter!"
    }else{
        label = "Non-cancer"
        suggestion="Penyakit kanker tidak terdeteksi."
    }
   
    return {label, suggestion };
  }catch (error) {
    throw new InputError(`Terjadi kesalahan input: ${error.message}`)
}
}
 
module.exports = predictClassification;