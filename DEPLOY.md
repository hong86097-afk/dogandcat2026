# 🚀 Deploy to GitHub + Streamlit Cloud — Step by Step

Follow these steps to put your project online with a public URL anyone can visit.

---

## ✅ Files checklist

Before starting, make sure you have these files in your project folder on your computer:

```
your-project-folder/
├── app.py                  ← from this package
├── pro_4.ipynb             ← your training notebook
├── requirements.txt        ← from this package
├── README.md               ← from this package
├── .gitignore              ← from this package
├── .gitattributes          ← from this package
│
├── W1.npy                  ← from running the notebook
├── b1.npy
├── W2.npy
├── b2.npy
├── W3.npy
└── b3.npy
```

**12 files total.** Do NOT include the Cat/, Dog/, Train/, Test/, or Validation/ folders — those would push too many files (GitHub limit is 100 per upload).

---

## Step 1 — Train the model and save weights (one time)

Open `pro_4.ipynb` and **Run All** cells. The last cells save the `.npy` weight files to your project folder. You should see 6 new files:

```
W1.npy  b1.npy  W2.npy  b2.npy  W3.npy  b3.npy
```

---

## Step 2 — Test the app locally first

Open VS Code terminal in your project folder and run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

A browser opens at `http://localhost:8501`. Upload a photo and confirm everything works. If it does, stop the app with `Ctrl+C` and continue.

---

## Step 3 — Tell Git who you are (first time only)

```bash
git config --global user.name "Your Name"
git config --global user.email "your@email.com"
```

Use the **same email** as your GitHub account.

---

## Step 4 — Initialize Git and commit

In your project folder:

```bash
git init
git add .
git status
```

`git status` will show you what's about to be committed. **Check the list carefully** — you should see ~12 files, NOT 200+. If you see hundreds of photo files, something's wrong with `.gitignore`.

Once the list looks right:

```bash
git commit -m "Initial commit: Cat vs Dog classifier with Streamlit app"
```

---

## Step 5 — Create the GitHub repository

1. Go to **https://github.com/new**
2. Repository name: `cat-dog-classifier`
3. Make it **Public**
4. **Do NOT** check "Add README" — you already have one
5. **Do NOT** check "Add .gitignore" — you already have one
6. Click **"Create repository"**

---

## Step 6 — Push your code to GitHub

GitHub will show you commands on the next page. Use:

```bash
git remote add origin https://github.com/YOUR_USERNAME/cat-dog-classifier.git
git branch -M main
git push -u origin main
```

Replace `YOUR_USERNAME` with your actual GitHub username.

**If GitHub asks for a password:** use a **Personal Access Token**, not your password. To make one:
- GitHub → Settings → Developer settings → Personal access tokens → Generate new token (classic)
- Check the **"repo"** box
- Click Generate
- Copy the token and paste it as the password

---

## Step 7 — Deploy to Streamlit Cloud

1. Go to **https://streamlit.io/cloud**
2. Sign in with your GitHub account
3. Click **"New app"**
4. Fill in:
   - **Repository:** `YOUR_USERNAME/cat-dog-classifier`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Click **"Deploy"**

Wait 2–3 minutes. Streamlit will install dependencies and start your app.

Your app will be live at:
```
https://YOUR_USERNAME-cat-dog-classifier-app-RANDOM.streamlit.app
```

Bookmark this URL — share it with classmates and your teacher!

---

## 🔄 Updating your project later

Whenever you make changes (fix a bug, retrain the model):

```bash
git add .
git commit -m "Describe what you changed"
git push
```

Streamlit Cloud will auto-redeploy in ~2 minutes.

---

## 🐛 Troubleshooting

### "Yowza, that's a lot of files" on GitHub upload
You're using the web upload interface and trying to upload Cat/Dog folders. Use Step 4–6 (terminal commands) instead — `.gitignore` will block the photos automatically.

### `ModuleNotFoundError: libGL.so.1` on Streamlit Cloud
The `requirements.txt` already uses `opencv-python-headless` which avoids this. If you see this error, double-check the line says `opencv-python-headless` (not `opencv-python`).

### App crashes with "Missing model files"
The `.npy` weight files didn't get pushed. Run `git status` to check, then `git add *.npy` and push again.

### "Authentication failed" on git push
Use a Personal Access Token instead of your password (see Step 6).

### Files showing as `.gitignore.txt` in Windows
Turn on file extensions in File Explorer (View → File name extensions), then rename to remove the `.txt`.

---

## 🎉 You're done!

Once deployed, your project has:
- ✅ A public GitHub repo your teacher can review
- ✅ A live web app classmates can try
- ✅ A professional setup that shows real engineering work

Good luck with your presentation! 🐾
