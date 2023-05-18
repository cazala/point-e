const fetch = require('node-fetch')
const FormData = require('form-data')
const fs = require('fs')

const TEXT_TO_MESH_SERVER = 'https://2218-181-13-71-243.sa.ngrok.io'

async function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms))
}

async function sendPrompt(prompt) {
  const res = await fetch(`${TEXT_TO_MESH_SERVER}/prompt?text=${prompt}`)
  if (res.ok) {
    const data = await res.json()
    return data.id
  } else {
    return null
  }
}

async function isReady(id) {
  const res = await fetch(`${TEXT_TO_MESH_SERVER}/files/${id}.ply`)
  const text = await res.text()
  return !text.includes('<!DOCTYPE html>')
}

async function getPly(id) {
  const res = await fetch(`${TEXT_TO_MESH_SERVER}/files/${id}.ply`)
  const fileStream = fs.createWriteStream(`./${id}.ply`);
  return new Promise((resolve, reject) => {
    res.body.pipe(fileStream);
    res.body.on("error", reject);
    fileStream.on("finish", () => resolve(true));
  });
}

async function uploadFile(path) {
  const body = new FormData()

  body.append('file', fs.createReadStream(path))
  body.append('key', Math.random().toString(16).slice(2))
  body.append('app', 'conversion')

  const res = await fetch("https://api.products.aspose.app/3d/api/v2/file", {
    body,
    method: "post"
  })

  if (res.ok) {
    const data = await res.json()
    return data.id
  } else {
    throw new Error(await res.text())
  }
}

async function convertFile(id) {
  const res = await fetch("https://api.products.aspose.app/3d/api/v2/conversion", {
    body: JSON.stringify({
      id,
      "format": "GLB",
      "compression": false
    }),
    headers: {
      'Content-Type': 'application/json'
    },
    method: "post"
  })

  if (res.ok) {
    const data = await res.json()
    return data.id
  } else {
    throw new Error(await res.text())
  }
}

async function getUrl(id) {
  const res = await fetch(`https://api.products.aspose.app/3d/api/v2/job-state?id=${id}`)
  if (res.ok) {
    const data = await res.json()
    if (data.state === "Processed") {
      return data.payload
    } else {
      return null
    }
  } else {
    throw new Error(await res.text())
  }
}

async function downloadFile(url, path) {
  const res = await fetch(url);
  const fileStream = fs.createWriteStream(path);
  return new Promise((resolve, reject) => {
    res.body.pipe(fileStream);
    res.body.on("error", reject);
    fileStream.on("finish", resolve);
  });
}

async function main() {
  const prompt = 'a red car'
  console.log(`Sending prompt...`)
  let promptId = await sendPrompt(prompt)
  while (!promptId) {
    await sleep(1000)
    console.log(`Pending...`)
    promptId = await sendPrompt(promptId)
  }
  console.log(promptId)
  console.log(`Get Ply...`)
  let success = await isReady(promptId)
  while (!success) {
    await sleep(1000)
    console.log(`Pending...`)
    success = await isReady(promptId)
  }
  await getPly(promptId)
  console.log(`Done...`)
  console.log('Uploading file...')
  const uploadId = await uploadFile(`${promptId}.ply`)
  console.log(`Done!`)
  console.log(`ID: ${uploadId}`)
  console.log(`Converting...`)
  await convertFile(uploadId)
  console.log(`Done!`)
  console.log('Getting url...')
  let url = await getUrl(uploadId)
  while (!url) {
    await sleep(1000)
    console.log(`Pending...`)
    url = await getUrl(uploadId)
  }
  console.log(`Done!`)
  console.log(`URL: ${url}`)
  console.log('Downloading file...')
  await downloadFile(url, `${promptId}.glb`)
  console.log(`Done!`)
}

main().catch(e => console.error('Error', e))