# ETF_comparison

Some little tricks to handle the 2 github accounts you have
This assumes that you have a proper .ssh

Instructions taken from here: https://stackoverflow.com/questions/21160774/github-error-key-already-in-use

# Default GitHub
Host github-GiuseppeFasanella
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_rsa

#My other GitHub
Host github-LifeIsComplicated
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_e25519


  NEXT:
  edit .git/config to point to:
  
  github-LifeIsComplicated:<gh_username>/<gh_reponame>.git

  instead of the usual