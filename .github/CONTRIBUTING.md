# Contribution Guidelines

Thank you for your interest in contributing to Uni-Dock! We value your time and effort, and aim to make the contribution process enjoyable and efficient. To ensure consistency and maintainability, please follow these guidelines when submitting your contributions.

## Troubleshooting

If you encountered problems using Uni-Dock, please refer to our GitHub [issue tracker](https://github.com/dptech-corp/Uni-Dock/issues), and check if it is a known problem.
If you found a bug, you can help us improve by [submitting a new issue](https://github.com/dptech-corp/Uni-Dock/issues/new) to our GitHub Repository. Provide a clear and concise title and description, including steps to reproduce the issue (if applicable) and any relevant context.
Even better, you can submit a Pull Request with a patch.

## Feature requests

We highly appreciate your contributions, and would like to help you crafting the changes and making contributions to the community.
If you would like to implement a new feature, please **submit a feature requesting issue with a proposal for your work first**.
This help fitting your ideas and work with the development road map well, coordinating our efforts, and avoiding duplication of work.

## Submitting a Pull Request

**Please fork your own copy of the Uni-Dock repository, and draft your changes there.** Once the modified codes work as expected, please submit a pull request to merge your contributions.

1. [Fork](https://docs.github.com/en/github/getting-started-with-github/fork-a-repo) the [Uni-Dock repository](https://github.com/dptech-corp/Uni-Dock). If you already had an existing fork, [sync](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork) the fork to keep your modification up-to-date.

2. Pull your forked repository, create a new git branch, and make your changes in it:

     ```shell
     git checkout -b bug-fix
     ```

3. Coding your patch and commit the changes.

4. Push your branch to GitHub:

    ```shell
    git push origin my-fix-branch
    ```

5. On GitHub, create a pull request (PR) from your bug-fix branch targeting `dptech-corp/Uni-Dock`.

6. After your pull request is merged, you can safely delete your branch and sync the changes from the main (upstream) repository:

- Delete the remote branch on GitHub either [through the GitHub web UI](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-branches-in-your-repository/deleting-and-restoring-branches-in-a-pull-request#deleting-a-branch-used-for-a-pull-request) or your local shell as follows:

    ```shell
    git push origin --delete my-fix-branch
    ```

- Check out the master branch:

    ```shell
    git checkout develop -f
    ```

- Delete the local branch:

    ```shell
    git branch -D my-fix-branch
    ```

- Update your master with the latest upstream version:

    ```shell
    git pull --ff upstream develop
    ```
