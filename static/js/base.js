$(function () {
    $.get('/get_model/', {}, function (data) {
        $('a[href="#model"]').parent().removeClass('active');
        var model_id = {
            'Vector': '#vs',
            'GeneralizedVector': '#gvs'
        }[data];
        $(model_id).addClass('active');
    });

    $('a[href="#search"]').on('click', function (event) {
        event.preventDefault();
        var search_form = $('#search');
        search_form.addClass('open');
        search_form.find('> form > input[type="text"]').focus();
    });

    $('a[href="#build"]').on('click', function (event) {
        event.preventDefault();
        var build_form = $('#build');
        build_form.addClass('open');
        build_form.find('> form > input[type="text"]').focus();
    });

    $('a[href="#model"]').on('click', function (event) {
        event.preventDefault();
        var model = $(this).data('model');
        $.get('/set_model/', {'model': model});
        $('a[href="#model"]').parent().removeClass('active');
        $(this).parent().addClass('active');
    });

    $('.dialog, .dialog button.close').on('click keyup', function (event) {
        if (event.target === this || event.target.className === 'close' || event.keyCode === 27) {
            $(this).removeClass('open');
        }
    });

    $('#search').find('> form').submit(function (event) {
        event.preventDefault();
        var search_form = $('#search');
        var query = search_form.find('> form > input[type="text"]')[0].value;
        $.get('/search/', {'q': query}, function (data) {
            $('#content').html(data);
        });
        search_form.removeClass('open');
        $('#suggested').removeClass('active');
        $('#evaluate').removeClass('active');
    });

    $('#build').find('> form').submit(function (event) {
        event.preventDefault();
        var build_form = $('#build');
        var path = build_form.find('> form > input[type="text"]')[0].value;
        $.get('/build/', {'path': path}, function (data) {
            var message = "";
            if (data < 0)
                message = '<strong>An error occurred during building.</strong> ' +
                    'It might have been due misspellings in the path, please check it... ' +
                    'Otherwise contact with system admin.';
            else
                message = 'System successfully built in <strong>' + data + '</strong> seconds.';
            $('#content').html('<div class="alert alert-info" role="alert">' + message + '</div>');
        });
        $('.nav-link').removeClass('active');
        build_form.removeClass('open');
    });

    $('a[href="#suggested"]').on('click', function (event) {
        $(this).parent().addClass('active')
    });
});