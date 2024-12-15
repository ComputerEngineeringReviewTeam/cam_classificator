document.addEventListener('DOMContentLoaded', () => {
    const logoutLink = document.getElementById('logout-link');
    if (!logoutLink)
        return;

    logoutLink.addEventListener('click', event => {
        event.preventDefault();
        fetch('/cam/auth/logout', {method: 'POST'})
            .then(r => {
                if (r.redirected) {
                    console.log('Logged out')
                    window.location.href = r.url
                } else {
                    window.location.href = "/index"
                }
            })
            .catch(error => console.log('Error:', error))
    })
})