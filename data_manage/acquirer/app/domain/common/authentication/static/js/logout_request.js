async function logoutRequest() {
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
}