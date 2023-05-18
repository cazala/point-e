const fetch = require('node-fetch')
const { Readable } = require('stream');
const { finished } = require('stream/promises');
const FormData = require('form-data')
const fs = require('fs')

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
      // poll
      console.log('Pending...')
      await new Promise(resolve => setTimeout(resolve, 1000))
      return getUrl(id)
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
  const path = './mesh.ply'
  console.log('Uploading file...')
  const id = await uploadFile(path)
  console.log(`Done!`)
  console.log(`ID: ${id}`)
  console.log(`Converting...`)
  await convertFile(id)
  console.log(`Done!`)
  console.log('Getting url...')
  const url = await getUrl(id)
  console.log(`Done!`)
  console.log(`URL: ${url}`)
  console.log('Downloading file...')
  await downloadFile(url, 'mesh.glb')
  console.log(`Done!`)
}

main().catch(e => console.error(e))