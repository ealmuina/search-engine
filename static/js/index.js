$(function () {
    $('a[href="#search"]').on('click', function (event) {
        event.preventDefault();
        var search_form = $('#search');
        search_form.addClass('open');
        search_form.find('> form > input[type="search"]').focus();
    });

    $('#search, #search button.close').on('click keyup', function (event) {
        if (event.target === this || event.target.className === 'close' || event.keyCode === 27) {
            $(this).removeClass('open');
        }
    });

    $('form').submit(function (event) {
        event.preventDefault();

        var search_form = $('#search');
        var query = search_form.find('> form > input[type="search"]')[0].value;

        $.get('/search/', {'q': query}, function (data) {
            $('#results').html(data);
        });

        search_form.removeClass('open');
    })
});